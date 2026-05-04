from __future__ import annotations

import csv
import io
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_TRAINING_CONFIG, MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH
from src.models.inference import (
    load_model_metadata,
    load_trained_pipeline,
    predict_texts,
)


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "bad_request",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class BatchRecord:
    row_number: int
    text: str


class SpamDetectionService:
    text_column_candidates = ("text", "email", "message", "content", "body")

    def __init__(self) -> None:
        self.repo_root = REPO_ROOT
        self.max_batch_size = 5000
        self.max_text_length = 20000
        self.top_signal_count = DEFAULT_TRAINING_CONFIG.top_signal_count
        self.model_source = "src.models.inference"
        self.model_status = "booting"
        self.model_name = "unavailable"
        self.pipeline: Any = None
        self.metadata: dict[str, Any] = {}
        self.load_errors: list[str] = []
        self.notes: list[str] = []
        self.refresh_model()

    def refresh_model(self) -> None:
        self.metadata = load_model_metadata()
        self.load_errors = []
        self.notes = []
        self.model_name = self.metadata.get("best_model", "unavailable")

        try:
            self.pipeline = load_trained_pipeline()
        except FileNotFoundError as exc:
            self.pipeline = None
            self.model_status = "missing-artifact"
            self.load_errors.append(str(exc))
            self.notes.append(
                "Model artifact chua ton tai. Hay train model truoc khi du doan: `python -m src.models.train`."
            )
            return
        except Exception as exc:
            self.pipeline = None
            self.model_status = "error"
            self.load_errors.append(f"Khong the load trained pipeline: {exc}")
            self.notes.append("Khong the nap model; kiem tra artifact, dependency va metadata.")
            return

        classifier = getattr(getattr(self.pipeline, "named_steps", {}), "get", lambda _name: None)("classifier")
        if self.model_name == "unavailable" and classifier is not None:
            self.model_name = type(classifier).__name__
        elif self.model_name == "unavailable":
            self.model_name = type(self.pipeline).__name__

        try:
            self._validate_loaded_pipeline()
        except FileNotFoundError as exc:
            self.pipeline = None
            self.model_status = "missing-artifact"
            self.load_errors.append(str(exc))
            self.notes.append("Contextual model artifact thieu hoac khong hop le; hay train lai model.")
            return
        except Exception as exc:
            self.pipeline = None
            self.model_status = "error"
            self.load_errors.append(f"Model artifact khong hop le: {exc}")
            self.notes.append("Khong the nap day du contextual encoder; kiem tra artifact va dependency.")
            return

        self.model_status = "ready"
        if not self.metadata:
            self.notes.append("Da load pipeline nhung chua co metadata JSON; metrics se hien thi han che.")

    def predict_text(self, text: str) -> dict[str, Any]:
        cleaned = self._clean_text(text)
        self._ensure_model_ready()

        try:
            result = predict_texts(
                [cleaned],
                pipeline=self.pipeline,
                metadata=self.metadata,
                top_n=self.top_signal_count,
            )[0]
        except FileNotFoundError as exc:
            self.refresh_model()
            raise self._prediction_unavailable_error(exc) from exc
        except Exception as exc:
            raise ApiError(
                f"Khong the du doan email hien tai: {exc}",
                status_code=500,
                code="prediction_failed",
            ) from exc

        payload = self._serialize_prediction(result)
        payload["text"] = cleaned
        payload["text_preview"] = self._preview_text(cleaned)
        return payload

    def predict_batch_texts(self, texts: Sequence[str], source: str = "json") -> dict[str, Any]:
        records, warnings = self._build_json_records(texts)
        self._ensure_model_ready()

        try:
            results = predict_texts(
                [record.text for record in records],
                pipeline=self.pipeline,
                metadata=self.metadata,
                top_n=self.top_signal_count,
            )
        except FileNotFoundError as exc:
            self.refresh_model()
            raise self._prediction_unavailable_error(exc) from exc
        except Exception as exc:
            raise ApiError(
                f"Khong the batch predict voi input hien tai: {exc}",
                status_code=500,
                code="batch_prediction_failed",
            ) from exc

        rows = [self._merge_batch_row(record, result) for record, result in zip(records, results)]
        return self._build_batch_payload(rows=rows, source=source, warnings=warnings)

    def predict_uploaded_file(self, filename: str, content: bytes) -> dict[str, Any]:
        records, warnings = self._parse_uploaded_records(filename=filename, content=content)
        self._ensure_model_ready()

        try:
            results = predict_texts(
                [record.text for record in records],
                pipeline=self.pipeline,
                metadata=self.metadata,
                top_n=self.top_signal_count,
            )
        except FileNotFoundError as exc:
            self.refresh_model()
            raise self._prediction_unavailable_error(exc) from exc
        except Exception as exc:
            raise ApiError(
                f"Khong the doc file va du doan: {exc}",
                status_code=500,
                code="upload_prediction_failed",
            ) from exc

        rows = [self._merge_batch_row(record, result) for record, result in zip(records, results)]
        return self._build_batch_payload(rows=rows, source=filename, warnings=warnings)

    def get_model_info(self) -> dict[str, Any]:
        if self.pipeline is None and MODEL_ARTIFACT_PATH.exists():
            self.refresh_model()

        best_model = self.metadata.get("best_model")
        validation_results = self.metadata.get("validation_results", {})
        best_validation = (
            validation_results.get(best_model, {})
            if isinstance(validation_results, Mapping) and isinstance(best_model, str)
            else {}
        )

        return {
            "status": self.model_status,
            "source": self.model_source,
            "model_name": self.model_name,
            "artifact_path": self._relative_path(MODEL_ARTIFACT_PATH),
            "artifact_exists": MODEL_ARTIFACT_PATH.exists(),
            "metadata_path": self._relative_path(MODEL_METADATA_PATH),
            "metadata_exists": MODEL_METADATA_PATH.exists(),
            "labels": self.metadata.get("class_names", ["ham", "spam"]),
            "supports_probabilities": self.pipeline is not None,
            "metrics": self._extract_metrics(self.metadata.get("test_metrics", {})),
            "validation_metrics": self._extract_metrics(best_validation),
            "metadata": {
                "trained_at": self.metadata.get("trained_at"),
                "dataset_summary": self.metadata.get("dataset_summary"),
                "config": self.metadata.get("config"),
                "artifacts": self.metadata.get("artifacts"),
                "best_model": best_model,
                "validation_results": validation_results,
            },
            "notes": self.notes,
            "load_errors": self.load_errors,
        }

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_status": self.model_status,
            "model_source": self.model_source,
            "model_name": self.model_name,
            "artifact_exists": MODEL_ARTIFACT_PATH.exists(),
        }

    def _ensure_model_ready(self) -> None:
        if self.pipeline is not None:
            return

        self.refresh_model()
        if self.pipeline is None:
            message = self.load_errors[0] if self.load_errors else "Model chua san sang."
            raise ApiError(
                message,
                status_code=503,
                code="model_unavailable",
                details={
                    "artifact_path": str(MODEL_ARTIFACT_PATH),
                    "metadata_path": str(MODEL_METADATA_PATH),
                },
            )

    def _prediction_unavailable_error(self, exc: Exception) -> ApiError:
        return ApiError(
            str(exc),
            status_code=503,
            code="model_unavailable",
            details={
                "artifact_path": str(MODEL_ARTIFACT_PATH),
                "metadata_path": str(MODEL_METADATA_PATH),
            },
        )

    def _validate_loaded_pipeline(self) -> None:
        if self.pipeline is None:
            return
        if getattr(self.pipeline, "model_family", "") != "contextual_transformer":
            return

        loader = getattr(self.pipeline, "_load_inference_components", None)
        if callable(loader):
            loader()
            return

        encoder_dir = Path(getattr(self.pipeline, "encoder_dir", ""))
        if not encoder_dir.exists():
            raise FileNotFoundError(f"Fine-tuned transformer directory not found at {encoder_dir}.")

    def _serialize_prediction(self, result: Mapping[str, Any]) -> dict[str, Any]:
        spam_probability = self._to_float(result.get("spam_probability"))
        ham_probability = self._to_float(result.get("ham_probability"))
        confidence = self._to_float(result.get("confidence"))
        decision_score = self._to_float(result.get("decision_score"))

        return {
            "label": result.get("label", "unknown"),
            "label_id": result.get("label_id"),
            "confidence": confidence,
            "decision_score": decision_score,
            "spam_probability": spam_probability,
            "ham_probability": ham_probability,
            "scores": {
                "spam": spam_probability,
                "ham": ham_probability,
            },
            "top_signals": result.get("top_signals", []),
            "model_name": result.get("model_name") or self.model_name,
            "model_source": self.model_source,
        }

    def _build_json_records(self, texts: Sequence[str]) -> tuple[list[BatchRecord], list[str]]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise ApiError(
                "Truong texts phai la mot danh sach chuoi.",
                status_code=400,
                code="invalid_payload",
            )

        records: list[BatchRecord] = []
        ignored = 0
        for index, value in enumerate(texts, start=1):
            text = "" if value is None else str(value)
            cleaned = text.strip()
            if not cleaned:
                ignored += 1
                continue
            records.append(BatchRecord(row_number=index, text=self._clean_text(cleaned)))

        warnings = []
        if ignored:
            warnings.append(f"Da bo qua {ignored} dong rong trong payload JSON.")

        self._ensure_batch_not_empty(records)
        return records, warnings

    def _parse_uploaded_records(self, filename: str, content: bytes) -> tuple[list[BatchRecord], list[str]]:
        if not filename:
            raise ApiError("Can cung cap ten file hop le.", code="missing_filename")

        extension = Path(filename).suffix.casefold()
        if extension not in {".csv", ".txt"}:
            raise ApiError(
                "Chi ho tro upload file CSV hoac TXT.",
                status_code=415,
                code="unsupported_file_type",
                details={"filename": filename},
            )

        text_content = self._decode_upload(content)
        if extension == ".txt":
            records = self._parse_txt_records(text_content)
            warnings = [
                "File TXT duoc xu ly theo tung dong khong rong; moi dong duoc xem nhu mot email."
            ]
        else:
            records, column_name = self._parse_csv_records(text_content)
            warnings = [f"Da doc noi dung tu cot '{column_name}' trong file CSV."]

        self._ensure_batch_not_empty(records)
        return records, warnings

    def _decode_upload(self, content: bytes) -> str:
        if not content:
            raise ApiError("File rong, khong co du lieu de du doan.", code="empty_file")
        if b"\x00" in content:
            raise ApiError(
                "Khong the giai ma file upload. Hay luu file voi UTF-8 hoac UTF-8 BOM.",
                status_code=400,
                code="decode_error",
            )

        for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                decoded = content.decode(encoding)
            except UnicodeDecodeError:
                continue
            if self._looks_like_text_content(decoded):
                return decoded

        raise ApiError(
            "Khong the giai ma file upload. Hay luu file voi UTF-8 hoac UTF-8 BOM.",
            status_code=400,
            code="decode_error",
        )

    def _parse_txt_records(self, text_content: str) -> list[BatchRecord]:
        records = []
        for line_number, line in enumerate(text_content.splitlines(), start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            records.append(BatchRecord(row_number=line_number, text=self._clean_text(cleaned)))
        return records

    def _parse_csv_records(self, text_content: str) -> tuple[list[BatchRecord], str]:
        dialect = self._detect_csv_dialect(text_content)
        rows = list(csv.reader(io.StringIO(text_content), dialect=dialect))
        if not rows:
            return [], "column_1"

        has_header = self._has_csv_header(text_content) or self._row_looks_like_header(rows[0])
        if has_header:
            fieldnames = [
                str(value).strip() or f"column_{index + 1}"
                for index, value in enumerate(rows[0])
            ]
            column_name = self._select_text_column(fieldnames)
            target_index = fieldnames.index(column_name) if column_name in fieldnames else 0
            records: list[BatchRecord] = []
            for row_number, row in enumerate(rows[1:], start=2):
                if not row:
                    continue
                value = row[target_index] if target_index < len(row) else self._first_non_empty(row)
                cleaned = str(value).strip()
                if cleaned:
                    records.append(BatchRecord(row_number=row_number, text=self._clean_text(cleaned)))
            if records:
                return records, column_name or fieldnames[0]

        records = []
        for row_number, row in enumerate(rows, start=1):
            if not row:
                continue
            value = self._first_non_empty(row)
            cleaned = str(value).strip()
            if cleaned:
                records.append(BatchRecord(row_number=row_number, text=self._clean_text(cleaned)))
        return records, "column_1"

    def _detect_csv_dialect(self, text_content: str) -> csv.Dialect:
        sample = text_content[:4096]
        try:
            return csv.Sniffer().sniff(sample)
        except csv.Error:
            return csv.excel

    def _has_csv_header(self, text_content: str) -> bool:
        sample = text_content[:4096]
        try:
            return csv.Sniffer().has_header(sample)
        except csv.Error:
            return False

    def _select_text_column(self, fieldnames: Sequence[str]) -> str:
        normalized_to_original = {
            self._normalize_column_name(fieldname): fieldname
            for fieldname in fieldnames
            if fieldname is not None
        }
        for candidate in self.text_column_candidates:
            match = normalized_to_original.get(candidate)
            if match:
                return match
        return next(iter(fieldnames), "")

    def _normalize_column_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _row_looks_like_header(self, row: Sequence[Any]) -> bool:
        normalized_values = {
            self._normalize_column_name(str(value))
            for value in row
            if str(value).strip()
        }
        return any(candidate in normalized_values for candidate in self.text_column_candidates)

    def _looks_like_text_content(self, value: str) -> bool:
        sample = value[:4096]
        if not sample:
            return False

        printable = sum(1 for char in sample if char.isprintable() or char in "\n\r\t")
        control = sum(1 for char in sample if ord(char) < 32 and char not in "\n\r\t")
        return control == 0 and (printable / len(sample)) >= 0.85

    def _ensure_batch_not_empty(self, records: Sequence[BatchRecord]) -> None:
        if not records:
            raise ApiError(
                "Khong tim thay dong text hop le de du doan.",
                status_code=400,
                code="empty_batch",
            )
        if len(records) > self.max_batch_size:
            raise ApiError(
                f"So dong vuot qua gioi han {self.max_batch_size}.",
                status_code=413,
                code="batch_too_large",
                details={"limit": self.max_batch_size},
            )

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            raise ApiError("Noi dung email khong duoc de trong.", code="empty_text")
        if len(cleaned) > self.max_text_length:
            raise ApiError(
                f"Noi dung vuot qua gioi han {self.max_text_length} ky tu.",
                status_code=413,
                code="text_too_large",
                details={"limit": self.max_text_length},
            )
        return cleaned

    def _merge_batch_row(self, record: BatchRecord, result: Mapping[str, Any]) -> dict[str, Any]:
        prediction = self._serialize_prediction(result)
        return {
            "row_number": record.row_number,
            "text": record.text,
            "text_preview": self._preview_text(record.text),
            "label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "scores": prediction.get("scores", {}),
            "spam_probability": prediction.get("spam_probability"),
            "ham_probability": prediction.get("ham_probability"),
            "top_signals": prediction.get("top_signals", []),
        }

    def _build_batch_payload(
        self,
        rows: list[dict[str, Any]],
        source: str,
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        spam_count = sum(1 for row in rows if row.get("label") == "spam")
        ham_count = sum(1 for row in rows if row.get("label") == "ham")
        return {
            "rows": rows,
            "summary": {
                "total": len(rows),
                "spam": spam_count,
                "ham": ham_count,
                "source": source,
            },
            "warnings": list(warnings),
        }

    def _extract_metrics(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}

        extracted = {}
        for source_key, target_key in (
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("f1_score", "f1"),
            ("roc_auc", "roc_auc"),
            ("auc", "roc_auc"),
        ):
            value = payload.get(source_key)
            numeric = self._to_float(value)
            if numeric is not None:
                extracted[target_key] = numeric

        for source_key in ("tp", "tn", "fp", "fn", "support"):
            value = payload.get(source_key)
            if isinstance(value, (int, float)):
                extracted[source_key] = int(value)

        return extracted

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        return None

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def _preview_text(self, text: str, max_length: int = 140) -> str:
        return text if len(text) <= max_length else f"{text[: max_length - 1]}..."

    def _first_non_empty(self, values: Sequence[Any]) -> Any:
        for value in values:
            if str(value).strip():
                return value
        return ""
