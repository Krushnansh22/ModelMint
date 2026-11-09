"""GET metrics routes (build guide section 4/10 Monitoring Service)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import models as db_models
from app.database.database import get_db
from app.security.auth import AuthContext, require_scopes_if_enabled
from app.services import monitoring as monitoring_service

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

_read = require_scopes_if_enabled("require_auth_for_reads", "monitoring:read")


@router.get("/metrics")
def get_metrics(
    model_id: str | None = Query(default=None),
    _ctx: AuthContext = Depends(_read),
    db: Session = Depends(get_db),
):
    if model_id:
        exists = db.query(db_models.Model).filter(db_models.Model.id == model_id).first()
        if exists is None:
            raise HTTPException(status_code=404, detail="model not found")
    return monitoring_service.compute_aggregate_metrics(db, model_id=model_id)


@router.get("/timeseries")
def get_timeseries(
    model_id: str | None = Query(default=None),
    limit: int = 200,
    _ctx: AuthContext = Depends(_read),
    db: Session = Depends(get_db),
):
    return monitoring_service.latency_timeseries(db, model_id=model_id, limit=limit)


@router.get("/logs")
def get_logs(
    model_id: str | None = Query(default=None),
    limit: int = 50,
    _ctx: AuthContext = Depends(_read),
    db: Session = Depends(get_db),
):
    logs = monitoring_service.recent_logs(db, model_id=model_id, limit=limit)
    return [
        {
            "id": log.id,
            "model_id": log.model_id,
            "timestamp": log.timestamp.isoformat(),
            "status": log.status,
            "latency_ms": log.latency_ms,
            "error_message": log.error_message,
            "api_key_id": log.api_key_id,
            "request_id": log.request_id,
            "task_type": log.task_type,
        }
        for log in logs
    ]
