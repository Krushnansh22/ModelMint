"""Upload, list, and get-details routes."""
from __future__ import annotations

import os
import re
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.models.schemas import ModelStatus
from app.services.analyzer import analyze_and_finalize
from app.services.api_generator import build_example_request

router = APIRouter(prefix="/api/models", tags=["models"])

STORAGE_DIR = os.environ.get("MODEL_STORAGE_DIR", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pkl", ".joblib", ".onnx"}


def _slugify(name: str, db: Session) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
    slug = base
    n = 2
    while db.query(db_models.Model).filter(db_models.Model.slug == slug).first() is not None:
        slug = f"{base}-{n}"
        n += 1
    return slug


@router.post("", status_code=201)
async def upload_model(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    model_id = str(uuid.uuid4())
    # Store by generated id, never trust the original filename for the path
    # on disk (build guide section 6: don't let the public API path be an
    # unsanitized filename).
    dest_path = os.path.join(STORAGE_DIR, f"{model_id}{ext}")

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    display_name = name or os.path.splitext(file.filename or "model")[0]
    slug = _slugify(display_name, db)

    model_row = db_models.Model(
        id=model_id,
        name=display_name,
        slug=slug,
        framework="unknown",
        file_path=dest_path,
        file_size_bytes=len(contents),
        status=ModelStatus.UPLOADING.value,
    )
    db.add(model_row)
    db.commit()
    db.refresh(model_row)

    model_row = analyze_and_finalize(db, model_row)
    return _serialize(model_row)


@router.get("")
def list_models(db: Session = Depends(get_db)):
    rows = db.query(db_models.Model).order_by(db_models.Model.created_at.desc()).all()
    return [_serialize(r) for r in rows]


@router.get("/{id_or_slug}")
def get_model(id_or_slug: str, db: Session = Depends(get_db)):
    row = _lookup(id_or_slug, db)
    payload = _serialize(row)
    payload["example_request"] = build_example_request(row)
    return payload


def _lookup(id_or_slug: str, db: Session) -> db_models.Model:
    row = (
        db.query(db_models.Model)
        .filter((db_models.Model.id == id_or_slug) | (db_models.Model.slug == id_or_slug))
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="model not found")
    return row


def _serialize(row: db_models.Model) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "slug": row.slug,
        "framework": row.framework,
        "model_type": row.model_type,
        "file_size_bytes": row.file_size_bytes,
        "input_schema": row.input_schema,
        "output_schema": row.output_schema,
        "schema_confidence": row.schema_confidence,
        "status": row.status,
        "failure_reason": row.failure_reason,
        "load_time_ms": row.load_time_ms,
        "metadata": row.metadata_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
