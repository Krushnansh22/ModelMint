"""Upload, list, get-details, and delete routes."""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import models as db_models
from app.database.database import get_db
from app.models.adapters.registry import supported_extensions
from app.models.registry import model_manager
from app.models.schemas import ModelStatus
from app.security.audit import record_audit
from app.security.auth import (
    AuthContext,
    require_scopes,
    require_scopes_if_enabled,
)
from app.security.rate_limit import upload_rate_limit
from app.services.analyzer import analyze_and_finalize
from app.services.api_generator import build_example_request
from app.services.storage import get_storage

router = APIRouter(prefix="/api/models", tags=["models"])

settings = get_settings()
STORAGE_DIR = settings.model_storage_dir
TEMP_DIR = os.path.join(STORAGE_DIR, "_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = supported_extensions()
CHUNK_SIZE = 1024 * 1024


def _slugify(name: str, db: Session) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
    slug = base
    n = 2
    while db.query(db_models.Model).filter(db_models.Model.slug == slug).first() is not None:
        slug = f"{base}-{n}"
        n += 1
    return slug


def _next_version(db: Session, family_id: str) -> int:
    current = (
        db.query(func.max(db_models.Model.version))
        .filter(db_models.Model.family_id == family_id)
        .scalar()
    )
    return int(current or 0) + 1


@router.post("", status_code=201)
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    family: str | None = Form(default=None),
    ctx: AuthContext = Depends(require_scopes("models:write")),
    _rate: AuthContext = Depends(upload_rate_limit),
    db: Session = Depends(get_db),
):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext or '(none)'}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    base = _lookup(family, db) if family else None
    if base is not None and base.family_id is None:
        # Backfill older rows that predate family ids.
        base.family_id = base.id
        db.commit()

    model_id = str(uuid.uuid4())
    storage_key = f"{model_id}{ext}"
    # Stream to a private temp file first; only hand it to the storage backend
    # after the size cap and checksum succeed.
    dest_path = os.path.join(TEMP_DIR, f"{model_id}{ext}")

    size = 0
    hasher = hashlib.sha256()
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds MAX_UPLOAD_MB={settings.max_upload_mb}",
                    )
                hasher.update(chunk)
                out.write(chunk)
    except HTTPException:
        _safe_remove(dest_path)
        raise
    except Exception as e:  # noqa: BLE001
        _safe_remove(dest_path)
        raise HTTPException(status_code=500, detail=f"failed to store upload: {e}") from e

    if size == 0:
        _safe_remove(dest_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    checksum = hasher.hexdigest()

    try:
        stored_path = get_storage().put_file(dest_path, storage_key)
    except Exception as e:  # noqa: BLE001
        _safe_remove(dest_path)
        raise HTTPException(status_code=500, detail=f"failed to persist upload: {e}") from e

    display_name = name or (base.name if base is not None else os.path.splitext(filename or "model")[0])
    if base is not None:
        family_id = base.family_id or base.id
        version = _next_version(db, family_id)
        base_slug = re.sub(r"-v\d+$", "", base.slug)
        slug = _slugify(f"{base_slug}-v{version}", db)
        is_active = False
        traffic_weight = 0.0
        promoted_at = None
    else:
        family_id = model_id
        version = 1
        slug = _slugify(display_name, db)
        is_active = True
        traffic_weight = 1.0
        promoted_at = datetime.now(timezone.utc)

    duplicate = (
        db.query(db_models.Model)
        .filter(db_models.Model.checksum == checksum, db_models.Model.deleted_at.is_(None))
        .first()
    )

    model_row = db_models.Model(
        id=model_id,
        name=display_name,
        slug=slug,
        framework="unknown",
        file_path=stored_path,
        file_size_bytes=size,
        checksum=checksum,
        artifact_format=ext.lstrip("."),
        storage_backend=settings.storage_backend,
        storage_key=storage_key,
        status=ModelStatus.UPLOADING.value,
        family_id=family_id,
        version=version,
        is_active=is_active,
        traffic_weight=traffic_weight,
        promoted_at=promoted_at,
    )
    db.add(model_row)
    db.commit()
    db.refresh(model_row)

    model_row = analyze_and_finalize(db, model_row)

    record_audit(
        db,
        action="model.upload",
        status="SUCCESS" if model_row.status != ModelStatus.FAILED.value else "FAILURE",
        actor_key_id=ctx.key_id,
        resource=model_row.id,
        request=request,
        detail=(
            f"file={filename} bytes={size} sha256={checksum[:12]} "
            f"family={family_id} version={version}"
        ),
    )

    payload = _serialize(model_row)
    if duplicate is not None:
        payload["duplicate_of"] = {
            "id": duplicate.id,
            "slug": duplicate.slug,
            "name": duplicate.name,
        }
    return payload


@router.get("")
def list_models(
    q: str | None = None,
    framework: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _ctx: AuthContext = Depends(require_scopes_if_enabled("require_auth_for_reads", "models:read")),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    query = db.query(db_models.Model).filter(db_models.Model.deleted_at.is_(None))
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(
            (db_models.Model.name.ilike(like)) | (db_models.Model.slug.ilike(like))
        )
    if framework:
        query = query.filter(db_models.Model.framework == framework)
    if status:
        query = query.filter(db_models.Model.status == status)

    rows = (
        query.order_by(db_models.Model.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_serialize(r) for r in rows]


@router.get("/{id_or_slug}")
def get_model(
    id_or_slug: str,
    _ctx: AuthContext = Depends(require_scopes_if_enabled("require_auth_for_reads", "models:read")),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)
    payload = _serialize(row)
    payload["example_request"] = build_example_request(row)
    return payload


@router.delete("/{id_or_slug}")
def delete_model(
    id_or_slug: str,
    request: Request,
    ctx: AuthContext = Depends(require_scopes("models:write")),
    db: Session = Depends(get_db),
):
    row = _lookup(id_or_slug, db)

    model_manager.evict(row.id)
    if row.storage_key:
        get_storage().delete(row.storage_key)
    _safe_remove(row.file_path)
    db.query(db_models.InferenceLog).filter(db_models.InferenceLog.model_id == row.id).delete()
    db.query(db_models.ApiKey).filter(db_models.ApiKey.model_id == row.id).update(
        {db_models.ApiKey.model_id: None}
    )
    db.delete(row)
    db.commit()

    record_audit(
        db,
        action="model.delete",
        actor_key_id=ctx.key_id,
        resource=row.id,
        request=request,
        detail=f"slug={row.slug}",
    )
    return {"deleted": True, "id": row.id, "slug": row.slug}


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


def _safe_remove(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _serialize(row: db_models.Model) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "slug": row.slug,
        "framework": row.framework,
        "model_type": row.model_type,
        "task_type": row.task_type,
        "artifact_format": row.artifact_format,
        "checksum": row.checksum,
        "file_size_bytes": row.file_size_bytes,
        "input_schema": row.input_schema,
        "output_schema": row.output_schema,
        "schema_confidence": row.schema_confidence,
        "status": row.status,
        "failure_reason": row.failure_reason,
        "load_time_ms": row.load_time_ms,
        "metadata": row.metadata_json,
        "version": row.version,
        "family_id": row.family_id,
        "is_active": bool(row.is_active),
        "traffic_weight": float(row.traffic_weight or 0.0),
        "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
