"""Model versioning, promotion/rollback, A/B traffic, and drift routes (M5)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database import models as db_models
from app.database.database import get_db
from app.security.audit import record_audit
from app.security.auth import (
    AuthContext,
    require_scopes,
    require_scopes_if_enabled,
)
from app.security.rate_limit import upload_rate_limit
from app.services.drift import compute_drift

router = APIRouter(prefix="/api/models", tags=["versions"])

_read = require_scopes_if_enabled("require_auth_for_reads", "models:read")
_write = require_scopes("models:write")
_monitor = require_scopes_if_enabled("require_auth_for_reads", "monitoring:read")

_SERVABLE = ("READY", "READY_UNVALIDATED")


class TrafficUpdate(BaseModel):
    weights: dict[str, float] = Field(
        ..., description="version id/slug -> relative traffic weight"
    )


@router.get("/{id_or_slug}/versions")
def list_versions(
    id_or_slug: str,
    _ctx: AuthContext = Depends(_read),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    return _family_summary(_family_versions(db, row))


@router.post("/{id_or_slug}/versions", status_code=201)
async def upload_version(
    id_or_slug: str,
    request: Request,
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    ctx: AuthContext = Depends(_write),
    _rate: AuthContext = Depends(upload_rate_limit),
    db: Session = Depends(get_db),
):
    # Reuse the hardened upload path (size cap, checksum, sandbox analysis).
    from app.api.models import upload_model

    return await upload_model(
        request=request,
        file=file,
        name=name,
        family=id_or_slug,
        ctx=ctx,
        _rate=_rate,
        db=db,
    )


@router.post("/{id_or_slug}/promote")
def promote_version(
    id_or_slug: str,
    request: Request,
    ctx: AuthContext = Depends(_write),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    if row.status not in _SERVABLE:
        raise HTTPException(status_code=409, detail=f"version is not servable (status={row.status})")

    versions = _family_versions(db, row)
    for version in versions:
        version.is_active = version.id == row.id
        version.traffic_weight = 1.0 if version.id == row.id else 0.0
    row.promoted_at = datetime.now(timezone.utc)
    db.commit()

    record_audit(
        db,
        action="model.promote",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
        detail=f"family={_family_id(row)} version={row.version}",
    )
    return _family_summary(versions, promoted=row.id)


@router.post("/{id_or_slug}/rollback")
def rollback_version(
    id_or_slug: str,
    request: Request,
    ctx: AuthContext = Depends(_write),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    versions = _family_versions(db, row)
    candidates = [
        version
        for version in versions
        if version.version < row.version and version.status in _SERVABLE
    ]
    if not candidates:
        raise HTTPException(status_code=409, detail="no earlier servable version to roll back to")

    target = max(candidates, key=lambda version: version.version)
    for version in versions:
        version.is_active = version.id == target.id
        version.traffic_weight = 1.0 if version.id == target.id else 0.0
    target.promoted_at = datetime.now(timezone.utc)
    db.commit()

    record_audit(
        db,
        action="model.rollback",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
        detail=f"family={_family_id(row)} from_version={row.version} to_version={target.version}",
    )
    return _family_summary(versions, rolled_back_to=target.id)


@router.post("/{id_or_slug}/traffic")
def set_traffic(
    id_or_slug: str,
    body: TrafficUpdate,
    request: Request,
    ctx: AuthContext = Depends(_write),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    versions = _family_versions(db, row)
    if not versions:
        raise HTTPException(status_code=404, detail="model family not found")

    by_key = {}
    for version in versions:
        by_key[version.id] = version
        by_key[version.slug] = version

    requested: dict[str, float] = {}
    for key, weight in body.weights.items():
        version = by_key.get(key)
        if version is None:
            raise HTTPException(status_code=400, detail=f"unknown version in this family: {key}")
        if version.status not in _SERVABLE:
            raise HTTPException(
                status_code=400,
                detail=f"version {version.slug} is not servable (status={version.status})",
            )
        if weight < 0 or weight != weight or weight in (float("inf"), float("-inf")):
            raise HTTPException(status_code=400, detail=f"invalid weight for {key}: {weight}")
        requested[version.id] = float(weight)

    total = sum(requested.values())
    if total <= 0:
        raise HTTPException(status_code=400, detail="at least one traffic weight must be > 0")

    for version in versions:
        weight = requested.get(version.id, 0.0) / total
        version.traffic_weight = round(weight, 6)
        version.is_active = version.traffic_weight > 0
        if version.is_active and version.promoted_at is None:
            version.promoted_at = datetime.now(timezone.utc)
    db.commit()

    record_audit(
        db,
        action="model.traffic",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
        detail=f"family={_family_id(row)} weights={ {v.slug: v.traffic_weight for v in versions} }",
    )
    return _family_summary(versions)


@router.get("/{id_or_slug}/drift")
def get_drift(
    id_or_slug: str,
    window: int = Query(default=100, ge=10, le=1000),
    threshold: float = Query(default=3.0, gt=0, le=20),
    _ctx: AuthContext = Depends(_monitor),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    return compute_drift(db, row, window=window, threshold=threshold)


def _lookup(id_or_slug: str, db: Session) -> db_models.Model:
    row = (
        db.query(db_models.Model)
        .filter(
            ((db_models.Model.id == id_or_slug) | (db_models.Model.slug == id_or_slug))
            & db_models.Model.deleted_at.is_(None)
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="model not found")
    return row


def _family_id(row: db_models.Model) -> str:
    return row.family_id or row.id


def _family_versions(db: Session, row: db_models.Model) -> list[db_models.Model]:
    family_id = _family_id(row)
    return (
        db.query(db_models.Model)
        .filter(
            or_(
                db_models.Model.family_id == family_id,
                db_models.Model.id == family_id,
            )
            & db_models.Model.deleted_at.is_(None)
        )
        .order_by(db_models.Model.version.desc())
        .all()
    )


def _family_summary(versions: list[db_models.Model], **extra) -> dict:
    payload = {
        "family_id": _family_id(versions[0]) if versions else None,
        "total": len(versions),
        "versions": [_serialize_version(version) for version in versions],
    }
    payload.update(extra)
    return payload


def _serialize_version(row: db_models.Model) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "slug": row.slug,
        "version": row.version,
        "family_id": row.family_id,
        "status": row.status,
        "framework": row.framework,
        "task_type": row.task_type,
        "is_active": bool(row.is_active),
        "traffic_weight": float(row.traffic_weight or 0.0),
        "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "checksum": row.checksum,
    }
