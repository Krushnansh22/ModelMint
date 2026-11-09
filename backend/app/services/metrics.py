"""Prometheus text exposition (dependency-free).

Kept intentionally small: counts are computed from the database on scrape.
That is fine for a demo/single-node deployment; a high-traffic deployment
should move to `prometheus_client` counters/gauges in-process.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import models as db_models


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(db: Session) -> str:
    lines: list[str] = []

    lines.append("# HELP modelmint_models_total Models by status")
    lines.append("# TYPE modelmint_models_total gauge")
    rows = (
        db.query(db_models.Model.status, func.count(db_models.Model.id))
        .filter(db_models.Model.deleted_at.is_(None))
        .group_by(db_models.Model.status)
        .all()
    )
    for status, count in rows:
        lines.append(f'modelmint_models_total{{status="{_escape(status or "unknown")}"}} {count}')

    lines.append("# HELP modelmint_inference_requests_total Inference requests by status")
    lines.append("# TYPE modelmint_inference_requests_total counter")
    rows = (
        db.query(db_models.InferenceLog.status, func.count(db_models.InferenceLog.id))
        .group_by(db_models.InferenceLog.status)
        .all()
    )
    for status, count in rows:
        lines.append(
            f'modelmint_inference_requests_total{{status="{_escape(status or "unknown")}"}} {count}'
        )

    lines.append("# HELP modelmint_inference_latency_ms_sum Total inference latency in ms")
    lines.append("# TYPE modelmint_inference_latency_ms_sum counter")
    total = db.query(func.coalesce(func.sum(db_models.InferenceLog.latency_ms), 0.0)).scalar() or 0.0
    lines.append(f"modelmint_inference_latency_ms_sum {float(total):.3f}")

    lines.append("# HELP modelmint_inference_latency_ms_count Number of inference requests")
    lines.append("# TYPE modelmint_inference_latency_ms_count counter")
    count = db.query(func.count(db_models.InferenceLog.id)).scalar() or 0
    lines.append(f"modelmint_inference_latency_ms_count {int(count)}")

    lines.append("# HELP modelmint_api_keys_active_total Non-revoked API keys")
    lines.append("# TYPE modelmint_api_keys_active_total gauge")
    active = (
        db.query(func.count(db_models.ApiKey.id))
        .filter(db_models.ApiKey.revoked_at.is_(None))
        .scalar()
        or 0
    )
    lines.append(f"modelmint_api_keys_active_total {int(active)}")

    return "\n".join(lines) + "\n"
