"""Append-only audit log for security-relevant events."""
from __future__ import annotations

from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from app.database import models as db_models
from app.security.utils import client_ip, get_request_id


def record_audit(
    db: Session,
    *,
    action: str,
    status: str = "SUCCESS",
    actor_key_id: Optional[str] = None,
    resource: Optional[str] = None,
    request: Optional[Request] = None,
    detail: Optional[str] = None,
) -> None:
    """Best-effort audit write: never let telemetry break the request."""
    entry = db_models.AuditLog(
        actor_key_id=actor_key_id,
        action=action,
        resource=resource,
        status=status,
        ip_address=client_ip(request),
        request_id=get_request_id(request),
        detail=detail,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
