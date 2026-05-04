from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.local import LocalProxy
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.api.routes import web_bp
    from src.api.services import ApiError, SpamDetectionService
else:
    from .routes import web_bp
    from .services import ApiError, SpamDetectionService


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=None,
    )
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "spam-lab-dev"),
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=int(os.getenv("SPAM_WEB_MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))),
    )
    app.config["SPAM_SERVICE"] = SpamDetectionService()

    app.register_blueprint(web_bp)
    _register_error_handlers(app)
    return app


_app_instance: Flask | None = None


def _get_app() -> Flask:
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        payload = {
            "ok": False,
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            },
        }
        if _wants_json_response():
            return jsonify(payload), error.status_code
        return (
            render_template(
                "error.html",
                error_title="Request error",
                error_code=error.code,
                message=error.message,
                details=error.details,
                model_info=app.config["SPAM_SERVICE"].get_model_info(),
            ),
            error.status_code,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error: RequestEntityTooLarge):
        api_error = ApiError(
            "File qua lon. Gioi han upload mac dinh la 4 MB.",
            status_code=413,
            code="file_too_large",
            details={"limit_bytes": app.config["MAX_CONTENT_LENGTH"]},
        )
        return handle_api_error(api_error)

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        payload = {
            "ok": False,
            "error": {
                "code": error.name.casefold().replace(" ", "_"),
                "message": error.description,
                "details": {"status_code": error.code},
            },
        }
        if _wants_json_response():
            return jsonify(payload), error.code
        return (
            render_template(
                "error.html",
                error_title=error.name,
                error_code=error.code,
                message=error.description,
                details={"status_code": error.code},
                model_info=app.config["SPAM_SERVICE"].get_model_info(),
            ),
            error.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error: Exception):
        payload = {
            "ok": False,
            "error": {
                "code": "internal_server_error",
                "message": "Co loi khong mong doi xay ra trong server.",
                "details": {"reason": str(error)},
            },
        }
        if _wants_json_response():
            return jsonify(payload), 500
        return (
            render_template(
                "error.html",
                error_title="Internal server error",
                error_code=500,
                message="Co loi khong mong doi xay ra trong server.",
                details={"reason": str(error)},
                model_info=app.config["SPAM_SERVICE"].get_model_info(),
            ),
            500,
        )


def _wants_json_response() -> bool:
    if request.path.startswith("/api/"):
        return True

    best_match = request.accept_mimetypes.best_match(["application/json", "text/html"])
    if not best_match:
        return False
    return (
        best_match == "application/json"
        and request.accept_mimetypes[best_match] >= request.accept_mimetypes["text/html"]
    )

app = LocalProxy(_get_app)


if __name__ == "__main__":
    _get_app().run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
