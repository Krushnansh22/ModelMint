"""Aggregate metrics computed from inference_logs on read (build guide
section 7: no separate running counters for MVP)."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import models as db_models


def compute_aggregate_metrics(db: Session, model_id: Optional[str] = None) -> dict:
    query = select(db_models.InferenceLog)
    if model_id:
        query = query.where(db_models.InferenceLog.model_id == model_id)

    rows = db.execute(query).scalars().all()
    total = len(rows)
    success = sum(1 for r in rows if r.status == "SUCCESS")
    failed = total - success
    latencies = sorted(r.latency_ms for r in rows if r.status == "SUCCESS")

    def _p95(values: list[float]) -> Optional[float]:
        if not values:
            return None
        idx = max(0, int(round(0.95 * (len(values) - 1))))
        return values[idx]

    return {
        "model_id": model_id,
        "total_requests": total,
        "success_count": success,
        "failed_count": failed,
        "success_rate": (success / total) if total else None,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "min_latency_ms": latencies[0] if latencies else None,
        "max_latency_ms": latencies[-1] if latencies else None,
        "p95_latency_ms": _p95(latencies),
    }


def recent_logs(db: Session, model_id: Optional[str] = None, limit: int = 50) -> list[db_models.InferenceLog]:
    query = select(db_models.InferenceLog).order_by(db_models.InferenceLog.timestamp.desc()).limit(limit)
    if model_id:
        query = query.where(db_models.InferenceLog.model_id == model_id)
    return list(db.execute(query).scalars().all())


def latency_timeseries(db: Session, model_id: Optional[str] = None, limit: int = 200) -> list[dict]:
    logs = recent_logs(db, model_id=model_id, limit=limit)
    return [
        {"timestamp": log.timestamp.isoformat(), "latency_ms": log.latency_ms, "status": log.status}
        for log in reversed(logs)
    ]
