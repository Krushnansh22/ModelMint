"""Generic inference route (build guide section 5, Option A) --
POST /api/models/{id}/predict."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.models.adapters.base import AdapterInferenceError, AdapterLoadError
from app.models.registry import model_manager
from app.services.api_generator import validate_payload_against_schema
from app.services.inference import dispatch_predict

router = APIRouter(prefix="/api/models", tags=["inference"])


@router.post("/{id_or_slug}/predict")
async def predict(id_or_slug: str, payload: dict, request: Request, db: Session = Depends(get_db)):
    row = (
        db.query(db_models.Model)
        .filter((db_models.Model.id == id_or_slug) | (db_models.Model.slug == id_or_slug))
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="model not found")

    if row.status not in ("READY", "READY_UNVALIDATED"):
        raise HTTPException(status_code=409, detail=f"model is not servable (status={row.status})")

    try:
        validate_payload_against_schema(row, payload)
    except AdapterInferenceError as e:
        _log(db, row.id, "FAILED", 0.0, str(e))
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for this model", "reason": str(e)},
        )

    try:
        adapter, was_cold_start = model_manager.get_or_load(row.id, row.framework, row.file_path)
    except AdapterLoadError as e:
        _log(db, row.id, "FAILED", 0.0, str(e))
        raise HTTPException(status_code=422, detail=f"model failed to (re)load: {e}")

    try:
        result, latency_ms = await dispatch_predict(adapter, payload)
    except AdapterInferenceError as e:
        _log(db, row.id, "FAILED", 0.0, str(e))
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for this model", "reason": str(e)},
        )

    _log(db, row.id, "SUCCESS", latency_ms, None)

    return {
        "model_id": row.id,
        "predictions": result.predictions,
        "output_shape": result.raw_output_shape,
        "latency_ms": round(latency_ms, 3),
        "cold_start": was_cold_start,
    }


def _log(db: Session, model_id: str, status: str, latency_ms: float, error_message: str | None) -> None:
    db.add(
        db_models.InferenceLog(
            model_id=model_id,
            status=status,
            latency_ms=latency_ms,
            error_message=error_message,
        )
    )
    db.commit()
