"""ModelMint backend entrypoint."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.api import inference as inference_api
from app.api import keys as keys_api
from app.api import meta as meta_api
from app.api import models as models_api
from app.api import monitoring as monitoring_api
from app.api import versions as versions_api
from app.config import get_settings
from app.database.database import SessionLocal, get_db, init_db
from app.models.adapters.base import (
    AdapterDetectionError,
    AdapterInferenceError,
    AdapterLoadError,
)
from app.security.auth import bootstrap_admin_key
from app.security.headers import security_headers_middleware
from app.services.metrics import render_prometheus

settings = get_settings()
logger = logging.getLogger("modelmint")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    if not settings.log_json:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())


_configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.makedirs(settings.model_storage_dir, exist_ok=True)
    init_db()

    db = SessionLocal()
    try:
        created = bootstrap_admin_key(db)
        if created:
            logger.warning(
                "Created bootstrap admin key from BOOTSTRAP_ADMIN_KEY "
                "(prefix=%s). Remove it from the environment after use.",
                created[:16],
            )
    finally:
        db.close()

    yield


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)

app.middleware("http")(security_headers_middleware)


@app.exception_handler(AdapterDetectionError)
def handle_detection_error(_request, exc: AdapterDetectionError):
    return JSONResponse(status_code=400, content={"detail": "Unsupported model format", "reason": str(exc)})


@app.exception_handler(AdapterLoadError)
def handle_load_error(_request, exc: AdapterLoadError):
    return JSONResponse(status_code=422, content={"detail": "Model failed to load", "reason": str(exc)})


@app.exception_handler(AdapterInferenceError)
def handle_inference_error(_request, exc: AdapterInferenceError):
    return JSONResponse(status_code=400, content={"detail": "Invalid input for this model", "reason": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    from sqlalchemy import text

    from app.database.database import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": str(e)})
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(db: Session = Depends(get_db)):
    return PlainTextResponse(
        render_prometheus(db),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


app.include_router(models_api.router)
app.include_router(inference_api.router)
app.include_router(inference_api.family_router)
app.include_router(versions_api.router)
app.include_router(monitoring_api.router)
app.include_router(keys_api.router)
app.include_router(meta_api.router)
