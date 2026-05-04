from __future__ import annotations

from collections.abc import Mapping
import sys
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, render_template, request, url_for

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.api.services import ApiError, SpamDetectionService
    from src.api.dataset_sampler import TestDatasetSampler
    from src.api.raw_dataset_sampler import RawDatasetSampler
else:
    from .services import ApiError, SpamDetectionService
    from .dataset_sampler import TestDatasetSampler
    from .raw_dataset_sampler import RawDatasetSampler


web_bp = Blueprint("web", __name__, template_folder="templates", static_folder="static")


def _service() -> SpamDetectionService:
    return current_app.config["SPAM_SERVICE"]


def _sampler() -> TestDatasetSampler:
    if "DATASET_SAMPLER" not in current_app.config:
        current_app.config["DATASET_SAMPLER"] = TestDatasetSampler()
    return current_app.config["DATASET_SAMPLER"]


def _raw_sampler() -> RawDatasetSampler:
    if "RAW_DATASET_SAMPLER" not in current_app.config:
        current_app.config["RAW_DATASET_SAMPLER"] = RawDatasetSampler()
    return current_app.config["RAW_DATASET_SAMPLER"]


def _dashboard_context(**overrides: Any) -> dict[str, Any]:
    sampler = _sampler()
    raw_sampler = _raw_sampler()
    context = {
        "model_info": _service().get_model_info(),
        "single_result": None,
        "batch_result": None,
        "single_error": None,
        "batch_error": None,
        "submitted_text": "",
        "dataset_available": sampler.is_available(),
        "dataset_samples": sampler.get_samples_for_display(10) if sampler.is_available() else [],
        "raw_dataset_available": raw_sampler.is_available(),
        "raw_dataset_samples": raw_sampler.get_samples_for_display(10) if raw_sampler.is_available() else [],
        "api_reference": [
            ("GET", url_for("web.api_health"), "Kiem tra service va trang thai model"),
            ("POST", url_for("web.api_predict"), "Du doan mot email voi JSON {'text': '...'}"),
            ("POST", url_for("web.api_batch_predict"), "Batch predict voi multipart file hoac JSON {'texts': [...]}"),
            ("GET", url_for("web.api_model_info"), "Thong tin model, metrics, artifact"),
            ("GET", url_for("web.api_dataset_samples"), "Lay danh sach mau test tu dataset"),
            ("GET", url_for("web.api_dataset_sample", email_id='<id>'), "Lay mot mau test cu the"),
            ("GET", url_for("web.api_raw_dataset_samples"), "Lay danh sach mau tu raw dataset"),
        ],
    }
    context.update(overrides)
    return context


@web_bp.get("/")
def home() -> str:
    return render_template("index.html", **_dashboard_context())


@web_bp.post("/predict")
def predict_page() -> str:
    submitted_text = request.form.get("text", "")
    context = _dashboard_context(submitted_text=submitted_text)

    try:
        context["single_result"] = _service().predict_text(submitted_text)
    except ApiError as exc:
        context["single_error"] = exc.message

    return render_template("index.html", **context)


@web_bp.post("/batch")
def batch_page() -> str:
    uploaded_file = request.files.get("file")
    context = _dashboard_context(submitted_text=request.form.get("text", ""))

    try:
        if uploaded_file is None or not uploaded_file.filename:
            raise ApiError("Hay chon mot file CSV hoac TXT truoc khi submit.", code="missing_file")
        context["batch_result"] = _service().predict_uploaded_file(
            filename=uploaded_file.filename,
            content=uploaded_file.read(),
        )
    except ApiError as exc:
        context["batch_error"] = exc.message

    return render_template("index.html", **context)


@web_bp.get("/model-info")
def model_info_page() -> str:
    return render_template(
        "model_info.html",
        model_info=_service().get_model_info(),
        api_reference=_dashboard_context()["api_reference"],
    )


@web_bp.get("/api/health")
def api_health():
    return jsonify({"ok": True, **_service().health_payload()})


@web_bp.get("/api/model-info")
def api_model_info():
    return jsonify({"ok": True, "model": _service().get_model_info()})


@web_bp.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, Mapping):
        raise ApiError(
            "Body phai la JSON object chua truong text.",
            status_code=400,
            code="invalid_json",
        )

    if "text" not in payload:
        raise ApiError(
            "JSON body thieu truong text.",
            status_code=400,
            code="missing_text",
        )

    prediction = _service().predict_text(str(payload.get("text", "")))
    return jsonify(
        {
            "ok": True,
            "prediction": prediction,
            "model": {
                "status": _service().model_status,
                "source": _service().model_source,
                "model_name": _service().model_name,
            },
        }
    )


@web_bp.post("/api/batch-predict")
def api_batch_predict():
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            raise ApiError(
                "Body JSON batch phai la object chua truong texts.",
                status_code=400,
                code="invalid_json",
            )
        texts = None
        for key in ("texts", "emails", "messages"):
            if key in payload:
                texts = payload.get(key)
                break
        result = _service().predict_batch_texts(texts=texts, source="json")
        return jsonify({"ok": True, **result})

    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        raise ApiError(
            "Yeu cau batch phai gui file CSV/TXT hoac JSON {'texts': [...]}",
            status_code=400,
            code="missing_batch_input",
        )

    result = _service().predict_uploaded_file(
        filename=uploaded_file.filename,
        content=uploaded_file.read(),
    )
    return jsonify({"ok": True, **result})


@web_bp.get("/api/dataset-samples")
def api_dataset_samples():
    """Get list of sample emails from test dataset."""
    sampler = _sampler()

    if not sampler.is_available():
        raise ApiError(
            "Test dataset khong kha dung. Hay dam bao file data/processed/test.csv ton tai.",
            status_code=404,
            code="dataset_not_found",
        )

    sampler.load_samples()
    samples = sampler.get_samples_for_display(20)

    return jsonify({
        "ok": True,
        "samples": samples,
        "total": sampler.total_samples,
        "spam_count": sampler.spam_count,
        "ham_count": sampler.ham_count,
    })


@web_bp.get("/api/dataset-sample/<email_id>")
def api_dataset_sample(email_id: str):
    """Get a specific sample email by ID."""
    sampler = _sampler()

    if not sampler.is_available():
        raise ApiError(
            "Test dataset khong kha dung.",
            status_code=404,
            code="dataset_not_found",
        )

    sampler.load_samples()
    sample = sampler.get_sample_by_id(email_id)

    if sample is None:
        raise ApiError(
            f"Khong tim thay email voi ID: {email_id}",
            status_code=404,
            code="sample_not_found",
        )

    return jsonify({
        "ok": True,
        "sample": {
            "email_id": sample.email_id,
            "source": sample.source,
            "label": sample.label,
            "label_name": sample.label_name,
            "text": sample.text,
            "text_length": sample.text_length,
        },
    })


@web_bp.post("/predict-sample")
def predict_sample_page():
    """Predict using a sample from test dataset."""
    email_id = request.form.get("email_id", "")
    label_filter = request.form.get("label_filter", "")

    sampler = _sampler()
    context = _dashboard_context(submitted_text="")

    try:
        if not sampler.is_available():
            raise ApiError("Test dataset khong kha dung.", code="dataset_not_found")

        sampler.load_samples()

        # Get sample by ID or random
        if email_id:
            sample = sampler.get_sample_by_id(email_id)
            if sample is None:
                raise ApiError(f"Khong tim thay email ID: {email_id}", code="sample_not_found")
        elif label_filter in ("0", "1"):
            sample = sampler.get_random_sample(label=int(label_filter))
            if sample is None:
                raise ApiError("Khong tim thay mau phu hop.", code="no_samples")
        else:
            sample = sampler.get_random_sample()
            if sample is None:
                raise ApiError("Dataset rong.", code="empty_dataset")

        # Run prediction
        context["single_result"] = _service().predict_text(sample.text)
        context["submitted_text"] = sample.text
        context["sample_info"] = {
            "email_id": sample.email_id,
            "source": sample.source,
            "actual_label": sample.label_name,
            "text_length": sample.text_length,
        }

    except ApiError as exc:
        context["single_error"] = exc.message

    return render_template("index.html", **context)


@web_bp.get("/api/raw-dataset-samples")
def api_raw_dataset_samples():
    """Get list of sample emails from raw dataset."""
    raw_sampler = _raw_sampler()

    if not raw_sampler.is_available():
        raise ApiError(
            "Raw dataset khong kha dung. Hay dam bao files data/raw/kaggle/spam_ham_dataset.csv hoac data/raw/enron/enron_spam.csv ton tai.",
            status_code=404,
            code="raw_dataset_not_found",
        )

    raw_sampler.load_samples()
    samples = raw_sampler.get_samples_for_display(20)
    stats_by_source = raw_sampler.get_stats_by_source()

    return jsonify({
        "ok": True,
        "samples": samples,
        "total": raw_sampler.total_samples,
        "spam_count": raw_sampler.spam_count,
        "ham_count": raw_sampler.ham_count,
        "available_sources": raw_sampler.get_available_sources(),
        "stats_by_source": stats_by_source,
    })


@web_bp.post("/predict-raw-sample")
def predict_raw_sample_page():
    """Predict using a sample from raw dataset."""
    label_filter = request.form.get("label_filter", "")
    source_filter = request.form.get("source_filter", "")

    raw_sampler = _raw_sampler()
    context = _dashboard_context(submitted_text="")

    try:
        if not raw_sampler.is_available():
            raise ApiError("Raw dataset khong kha dung.", code="raw_dataset_not_found")

        raw_sampler.load_samples()

        # Get sample with filters
        label = int(label_filter) if label_filter in ("0", "1") else None
        source = source_filter if source_filter else None

        sample = raw_sampler.get_random_sample(label=label, source=source)

        if sample is None:
            raise ApiError("Khong tim thay mau phu hop voi filter.", code="no_samples")

        # Run prediction
        context["single_result"] = _service().predict_text(sample.text)
        context["submitted_text"] = sample.text
        context["sample_info"] = {
            "email_id": sample.email_id,
            "source": sample.source,
            "actual_label": sample.label_name,
            "text_length": sample.text_length,
            "is_raw": True,
        }

    except ApiError as exc:
        context["single_error"] = exc.message

    return render_template("index.html", **context)
