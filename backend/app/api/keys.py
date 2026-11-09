"""API-key management routes (admin scope only)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import models as db_models
from app.database.database import get_db
from app.security.audit import record_audit
from app.security.auth import (
    AuthContext,
    create_api_key,
    require_scopes,
    revoke_api_key,
    rotate_api_key,
)

router = APIRouter(prefix="/api/keys", tags=["api-keys"])


class ApiKeyCreate(BaseModel):
    model_config = {"protected_namespaces": ()}

    name: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(default_factory=lambda: ["predict"])
    model_id: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


def _serialize(row: db_models.ApiKey) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "prefix": row.prefix,
        "scopes": row.scopes or [],
        "model_id": row.model_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
        "request_count": row.request_count or 0,
    }


@router.post("", status_code=201)
def create_key(
    payload: ApiKeyCreate,
    request: Request,
    ctx: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_db),
):
    if payload.model_id:
        exists = (
            db.query(db_models.Model)
            .filter((db_models.Model.id == payload.model_id) | (db_models.Model.slug == payload.model_id))
            .first()
        )
        if exists is None:
            raise HTTPException(status_code=404, detail="model not found")
        model_id = exists.id
    else:
        model_id = None

    try:
        row, plaintext = create_api_key(
            db,
            name=payload.name,
            scopes=payload.scopes,
            model_id=model_id,
            expires_in_days=payload.expires_in_days,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record_audit(
        db,
        action="key.create",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
        detail=f"scopes={','.join(row.scopes or [])}",
    )

    body = _serialize(row)
    # Plaintext is returned exactly once; it is never persisted or logged.
    body["key"] = plaintext
    return body


@router.get("")
def list_keys(
    ctx: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_db),
):
    rows = db.query(db_models.ApiKey).order_by(db_models.ApiKey.created_at.desc()).all()
    return [_serialize(r) for r in rows]


@router.delete("/{key_id}")
def revoke_key(
    key_id: str,
    request: Request,
    ctx: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_db),
):
    row = db.query(db_models.ApiKey).filter(db_models.ApiKey.id == key_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="api key not found")
    if row.id == ctx.key_id:
        raise HTTPException(status_code=400, detail="cannot revoke the key used for this request")
    if row.revoked_at is not None:
        return _serialize(row)

    revoke_api_key(db, row)
    record_audit(
        db,
        action="key.revoke",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
    )
    return _serialize(row)


@router.post("/{key_id}/rotate")
def rotate_key(
    key_id: str,
    request: Request,
    ctx: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_db),
):
    row = db.query(db_models.ApiKey).filter(db_models.ApiKey.id == key_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="api key not found")
    if row.revoked_at is not None:
        raise HTTPException(status_code=400, detail="cannot rotate a revoked key")

    new_row, plaintext = rotate_api_key(db, row)
    record_audit(
        db,
        action="key.rotate",
        actor_key_id=ctx.key_id,
        resource=new_row.id,
        request=request,
        detail=f"rotated_from={row.id}",
    )
    body = _serialize(new_row)
    body["key"] = plaintext
    return body


@router.get("/audit")
def list_audit_logs(
    limit: int = 100,
    ctx: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 500))
    rows = (
        db.query(db_models.AuditLog)
        .order_by(db_models.AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "actor_key_id": r.actor_key_id,
            "action": r.action,
            "resource": r.resource,
            "status": r.status,
            "ip_address": r.ip_address,
            "request_id": r.request_id,
            "detail": r.detail,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in rows
    ]
