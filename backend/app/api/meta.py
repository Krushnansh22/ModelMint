"""Public metadata routes: supported formats and non-secret runtime config."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.models.adapters.registry import list_formats

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/formats")
def get_formats():
    return {"formats": list_formats()}


@router.get("/config")
def get_public_config():
    """Safe, non-secret flags the frontend uses to decide what to show."""
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "require_auth_for_predict": settings.require_auth_for_predict,
        "require_auth_for_reads": settings.require_auth_for_reads,
        "max_upload_mb": settings.max_upload_mb,
    }
