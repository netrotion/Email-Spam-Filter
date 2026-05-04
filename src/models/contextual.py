from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.preprocessing.text import clean_email_text


def _require_transformer_dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `torch`. Install project requirements before training the contextual model."
        ) from exc

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `transformers`. Install project requirements before training the contextual model."
        ) from exc

    return torch, DataLoader, AutoTokenizer, AutoModelForSequenceClassification


def _major_minor_version(value: str) -> tuple[int, int] | None:
    parts = str(value).split(".")[:2]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


class _EncodedTextDataset:
    def __init__(self, texts, labels=None):
        self.texts = [clean_email_text(str(text or "")) for text in texts]
        self.labels = None if labels is None else [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["labels"] = int(self.labels[index])
        return item


@dataclass
class ContextualTransformerSpamPipeline:
    model_name: str
    encoder_dir: Path
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    freeze_base_model: bool = False
    warmup_ratio: float = 0.10
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 1
    use_class_weights: bool = True
    random_state: int = 42
    model_family: str = field(default="contextual_transformer", init=False)
    supports_explanations: bool = field(default=False, init=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.encoder_dir = Path(self.encoder_dir)

    def fit(self, texts, labels, **kwargs):
        (
            torch,
            DataLoader,
            AutoTokenizer,
            AutoModelForSequenceClassification,
        ) = _require_transformer_dependencies()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            torch_dtype=torch.float32,
        )
        if self.freeze_base_model:
            self._freeze_base_parameters(model)
        model.float()
        model.to(device)

        train_labels = [int(label) for label in labels]
        train_dataset = _EncodedTextDataset(texts, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self._build_collate_fn(tokenizer=tokenizer, torch=torch),
        )

        validation_texts = kwargs.get("validation_texts") or []
        validation_labels = kwargs.get("validation_labels") or []
        validation_loader = None
        if validation_texts and validation_labels:
            validation_dataset = _EncodedTextDataset(validation_texts, validation_labels)
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self._build_collate_fn(tokenizer=tokenizer, torch=torch),
            )

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        gradient_accumulation_steps = max(int(self.gradient_accumulation_steps), 1)
        total_update_steps = max(ceil(len(train_loader) / gradient_accumulation_steps) * self.epochs, 1)
        warmup_steps = int(total_update_steps * max(self.warmup_ratio, 0.0))
        scheduler = self._build_linear_warmup_scheduler(
            torch=torch,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_update_steps=total_update_steps,
        )
        class_weights = self._build_class_weights(train_labels, torch=torch, device=device) if self.use_class_weights else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        track_best_checkpoint = validation_loader is not None
        best_checkpoint_path = self.encoder_dir.with_name(f"{self.encoder_dir.name}_best_state.pt")
        if track_best_checkpoint and best_checkpoint_path.exists():
            best_checkpoint_path.unlink()
        best_score = float("-inf")
        epochs_without_improvement = 0
        validation_batches = len(validation_loader) if validation_loader is not None else 0
        print(
            "Contextual training started: "
            f"model={self.model_name}, device={device}, "
            f"train_examples={len(train_dataset)}, validation_examples={len(validation_texts)}, "
            f"train_batches={len(train_loader)}, validation_batches={validation_batches}, "
            f"epochs={self.epochs}, gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"optimizer_steps={total_update_steps}, max_grad_norm=1.0",
            flush=True,
        )
        log_every = max(len(train_loader) // 10, 1)
        for epoch_index in range(self.epochs):
            epoch_started_at = perf_counter()
            running_loss = 0.0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                labels_tensor = batch["labels"].to(device)
                model_inputs = self._model_inputs_from_batch(batch=batch, device=device)
                outputs = model(**model_inputs)
                logits = outputs.logits.float()
                batch_loss = loss_fn(logits, labels_tensor)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError(
                        "Non-finite transformer training loss detected. "
                        "Restart the runtime and train with the current stable configuration."
                    )
                running_loss += float(batch_loss.detach().cpu())
                loss = batch_loss / gradient_accumulation_steps
                loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            "Non-finite transformer gradient detected. "
                            "Restart the runtime and train with the current stable configuration."
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if step == 1 or step % log_every == 0 or step == len(train_loader):
                    average_loss = running_loss / step
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch_index + 1}/{self.epochs} "
                        f"step {step}/{len(train_loader)} "
                        f"loss={average_loss:.4f} lr={current_lr:.2e}",
                        flush=True,
                    )

            elapsed_seconds = perf_counter() - epoch_started_at
            if validation_loader is None:
                print(
                    f"Epoch {epoch_index + 1}/{self.epochs} finished "
                    f"in {elapsed_seconds / 60:.1f} min loss={running_loss / len(train_loader):.4f}",
                    flush=True,
                )
                continue

            validation_score = self._evaluate_f1(
                model=model,
                dataloader=validation_loader,
                torch=torch,
                device=device,
            )
            if validation_score > best_score:
                best_score = validation_score
                if track_best_checkpoint:
                    torch.save(model.state_dict(), best_checkpoint_path)
                epochs_without_improvement = 0
                print(
                    f"Epoch {epoch_index + 1}/{self.epochs} finished "
                    f"in {elapsed_seconds / 60:.1f} min loss={running_loss / len(train_loader):.4f} "
                    f"validation_f1={validation_score:.4f} best_f1={best_score:.4f} checkpoint=updated",
                    flush=True,
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"Epoch {epoch_index + 1}/{self.epochs} finished "
                    f"in {elapsed_seconds / 60:.1f} min loss={running_loss / len(train_loader):.4f} "
                    f"validation_f1={validation_score:.4f} best_f1={best_score:.4f} "
                    f"epochs_without_improvement={epochs_without_improvement}",
                    flush=True,
                )
                if (
                    self.early_stopping_patience >= 0
                    and epochs_without_improvement > self.early_stopping_patience
                ):
                    print("Early stopping triggered.", flush=True)
                    break

        if track_best_checkpoint and best_checkpoint_path.exists():
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            best_checkpoint_path.unlink()

        print("Saving contextual model artifacts...", flush=True)
        self._save_finetuned_model(model=model, tokenizer=tokenizer)
        print("Contextual training finished.", flush=True)
        self._tokenizer = None
        self._model = None
        self._device = None
        return self

    def predict(self, texts) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return probabilities.argmax(axis=1)

    def predict_proba(self, texts) -> np.ndarray:
        logits = self._predict_logits(texts)
        return self._stable_softmax(logits)

    def decision_function(self, texts) -> np.ndarray:
        logits = self._predict_logits(texts)
        return logits[:, 1] - logits[:, 0]

    def _stable_softmax(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 2), dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=1e4, neginf=-1e4)
        values = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(values, -745.0, 709.0))
        sums = exp_values.sum(axis=1, keepdims=True)
        probabilities = exp_values / np.maximum(sums, np.finfo(np.float64).tiny)
        return probabilities.astype(float)

    def _predict_logits(self, texts) -> np.ndarray:
        torch, DataLoader, _auto_tokenizer, _auto_model = _require_transformer_dependencies()
        tokenizer, model, device = self._load_inference_components()
        dataset = _EncodedTextDataset(texts)
        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self._build_collate_fn(tokenizer=tokenizer, torch=torch),
        )

        model.eval()
        logits_list: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in dataloader:
                outputs = model(**self._model_inputs_from_batch(batch=batch, device=device))
                logits_list.append(outputs.logits.detach().cpu().numpy())

        if not logits_list:
            return np.empty((0, 2), dtype=float)
        return np.vstack(logits_list).astype(float)

    def _load_inference_components(self):
        (
            torch,
            _data_loader,
            AutoTokenizer,
            AutoModelForSequenceClassification,
        ) = _require_transformer_dependencies()
        self._validate_transformers_artifact_version()
        if self._tokenizer is not None and self._model is not None and self._device is not None:
            return self._tokenizer, self._model, self._device

        if not self.encoder_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned transformer directory not found at {self.encoder_dir}."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(self.encoder_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(self.encoder_dir))
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return tokenizer, model, device

    def _validate_transformers_artifact_version(self) -> None:
        config_path = self.encoder_dir / "config.json"
        if not config_path.exists():
            return

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        expected_version = payload.get("transformers_version")
        if not expected_version:
            return

        try:
            installed_version = version("transformers")
        except PackageNotFoundError:
            return

        expected_major_minor = _major_minor_version(str(expected_version))
        installed_major_minor = _major_minor_version(installed_version)
        if expected_major_minor is None or installed_major_minor is None:
            return

        if installed_major_minor != expected_major_minor:
            raise RuntimeError(
                "Transformer artifact version mismatch: "
                f"encoder was saved with transformers {expected_version}, "
                f"but the current environment has transformers {installed_version}. "
                "Install the pinned project requirements before inference."
            )

    def _save_finetuned_model(self, *, model, tokenizer) -> None:
        temp_dir = self.encoder_dir.with_name(f"{self.encoder_dir.name}_tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        tokenizer.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir, safe_serialization=True)

        if self.encoder_dir.exists():
            shutil.rmtree(self.encoder_dir)
        temp_dir.replace(self.encoder_dir)

    def _build_collate_fn(self, *, tokenizer, torch):
        def collate(examples):
            batch = tokenizer(
                [example["text"] for example in examples],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            if "labels" in examples[0]:
                batch["labels"] = torch.tensor(
                    [example["labels"] for example in examples],
                    dtype=torch.long,
                )
            return batch

        return collate

    def _model_inputs_from_batch(self, *, batch, device) -> dict[str, Any]:
        return {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

    def _build_class_weights(self, labels: list[int], *, torch, device):
        counts = Counter(int(label) for label in labels)
        total = sum(counts.values())
        if not total:
            return None
        weights = [
            total / (2.0 * max(counts.get(label_id, 0), 1))
            for label_id in (0, 1)
        ]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _build_linear_warmup_scheduler(self, *, torch, optimizer, warmup_steps: int, total_update_steps: int):
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(warmup_steps, 1))
            remaining_steps = total_update_steps - current_step
            decay_steps = max(total_update_steps - warmup_steps, 1)
            return max(0.0, float(remaining_steps) / float(decay_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _freeze_base_parameters(self, model) -> None:
        base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
        if base_model is None:
            return
        for parameter in base_model.parameters():
            parameter.requires_grad = False

    def _evaluate_f1(self, *, model, dataloader, torch, device) -> float:
        labels: list[int] = []
        predictions: list[int] = []
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                logits = model(**self._model_inputs_from_batch(batch=batch, device=device)).logits
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
                labels.extend(batch["labels"].tolist())
        return float(f1_score(labels, predictions, zero_division=0))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_model"] = None
        state["_device"] = None
        return state
