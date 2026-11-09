"""Generic inference routes (build guide section 5, Option A).

- `POST /api/models/{id}/predict` serves one explicit model version.
- `POST /api/families/{family_id}/predict` serves the family alias, choosing a
  version by its A/B traffic weight.

M1 adds API-key auth, model-bound key enforcement, per-key rate limiting, and
request metadata on inference logs. M5 adds version attribution and a small
input summary for drift detection.
"""
from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database import models as db_models
from app.database.database import get_db
from app.models.adapters.base import AdapterInferenceError, AdapterLoadError
from app.models.registry import model_manager
from app.security.auth import (
    AuthContext,
    ensure_model_binding,
    require_scopes_if_enabled,
)
from app.security.rate_limit import predict_rate_limit
from app.security.utils import client_ip, get_request_id
from app.services.api_generator import validate_payload_against_schema
from app.services.drift import compute_input_summary, summarize_batch
from app.services.inference import dispatch_predict, dispatch_predict_batch
from app.services.storage import ensure_local_path

router = APIRouter(prefix="/api/models", tags=["inference"])
family_router = APIRouter(prefix="/api/families", tags=["inference"])

_SERVABLE = ("READY", "READY_UNVALIDATED")
_predict_auth = require_scopes_if_enabled("require_auth_for_predict", "predict")


@router.post("/{id_or_slug}/predict")
async def predict(
    id_or_slug: str,
    payload: dict,
    request: Request,
    ctx: AuthContext = Depends(_predict_auth),
    _rate: AuthContext = Depends(predict_rate_limit),
    db: Session = Depends(get_db),
):
    row = _lookup_servable(db, id_or_slug)
    return await _predict_row(row, payload, request, ctx, db)


@router.post("/{id_or_slug}/predict/batch")
async def predict_batch(
    id_or_slug: str,
    payload: dict,
    request: Request,
    ctx: AuthContext = Depends(_predict_auth),
    _rate: AuthContext = Depends(predict_rate_limit),
    db: Session = Depends(get_db),
):
    row = _lookup_servable(db, id_or_slug)
    return await _predict_batch_row(row, payload, request, ctx, db)


@family_router.post("/{family_id}/predict")
async def predict_family(
    family_id: str,
    payload: dict,
    request: Request,
    ctx: AuthContext = Depends(_predict_auth),
    _rate: AuthContext = Depends(predict_rate_limit),
    db: Session = Depends(get_db),
):
    row = _select_version(db, family_id)
    return await _predict_row(row, payload, request, ctx, db)


@family_router.post("/{family_id}/predict/batch")
async def predict_family_batch(
    family_id: str,
    payload: dict,
    request: Request,
    ctx: AuthContext = Depends(_predict_auth),
    _rate: AuthContext = Depends(predict_rate_limit),
    db: Session = Depends(get_db),
):
    row = _select_version(db, family_id)
    return await _predict_batch_row(row, payload, request, ctx, db)


async def _predict_row(
    row: db_models.Model,
    payload: dict,
    request: Request,
    ctx: AuthContext,
    db: Session,
) -> dict:
    ensure_model_binding(ctx, row.id)
    summary = compute_input_summary(payload, row.task_type)

    try:
        validate_payload_against_schema(row, payload)
    except AdapterInferenceError as e:
        _log(db, row, "FAILED", 0.0, str(e), ctx, request, summary)
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for this model", "reason": str(e)},
        )

    try:
        adapter, was_cold_start = model_manager.get_or_load(
            row.id, row.framework, ensure_local_path(row)
        )
    except AdapterLoadError as e:
        _log(db, row, "FAILED", 0.0, str(e), ctx, request, summary)
        raise HTTPException(status_code=422, detail=f"model failed to (re)load: {e}")

    try:
        result, latency_ms = await dispatch_predict(adapter, payload)
    except AdapterInferenceError as e:
        _log(db, row, "FAILED", 0.0, str(e), ctx, request, summary)
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for this model", "reason": str(e)},
        )

    _log(db, row, "SUCCESS", latency_ms, None, ctx, request, summary)

    return {
        "model_id": row.id,
        "family_id": row.family_id,
        "version": row.version,
        "task_type": row.task_type or "tabular",
        "predictions": result.predictions,
        "output_shape": result.raw_output_shape,
        "latency_ms": round(latency_ms, 3),
        "cold_start": was_cold_start,
    }


async def _predict_batch_row(
    row: db_models.Model,
    payload: dict,
    request: Request,
    ctx: AuthContext,
    db: Session,
) -> dict:
    ensure_model_binding(ctx, row.id)

    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise HTTPException(
            status_code=400,
            detail={"message": "payload must contain a non-empty 'instances' list"},
        )
    if len(instances) > 1000:
        raise HTTPException(status_code=400, detail="batch size exceeds 1000 instances")

    summary = summarize_batch(instances, row.task_type)
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise HTTPException(status_code=400, detail=f"instance {index} must be an object")
        try:
            validate_payload_against_schema(row, instance)
        except AdapterInferenceError as e:
            _log(db, row, "FAILED", 0.0, f"batch instance {index}: {e}", ctx, request, summary)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid input for this model",
                    "instance_index": index,
                    "reason": str(e),
                },
            )

    try:
        adapter, was_cold_start = model_manager.get_or_load(
            row.id, row.framework, ensure_local_path(row)
        )
    except AdapterLoadError as e:
        _log(db, row, "FAILED", 0.0, str(e), ctx, request, summary)
        raise HTTPException(status_code=422, detail=f"model failed to (re)load: {e}")

    try:
        results, latency_ms = await dispatch_predict_batch(adapter, instances)
    except AdapterInferenceError as e:
        _log(db, row, "FAILED", 0.0, str(e), ctx, request, summary)
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for this model", "reason": str(e)},
        )

    _log(db, row, "SUCCESS", latency_ms, None, ctx, request, summary)

    return {
        "model_id": row.id,
        "family_id": row.family_id,
        "version": row.version,
        "task_type": row.task_type or "tabular",
        "count": len(results),
        "predictions": [r.predictions for r in results],
        "latency_ms": round(latency_ms, 3),
        "cold_start": was_cold_start,
    }


def _lookup_servable(db: Session, id_or_slug: str) -> db_models.Model:
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
    if row.status not in _SERVABLE:
        raise HTTPException(status_code=409, detail=f"model is not servable (status={row.status})")
    return row


def _select_version(db: Session, family_id: str) -> db_models.Model:
    rows = (
        db.query(db_models.Model)
        .filter(
            or_(
                db_models.Model.family_id == family_id,
                db_models.Model.id == family_id,
            )
            & db_models.Model.deleted_at.is_(None)
            & db_models.Model.status.in_(_SERVABLE)
        )
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="no servable version in this model family")

    active = [row for row in rows if row.is_active and (row.traffic_weight or 0.0) > 0]
    if active:
        weights = [float(row.traffic_weight) for row in active]
        return random.choices(active, weights=weights, k=1)[0]

    # No explicit traffic split: serve the newest servable version.
    return max(rows, key=lambda row: row.version)


def _log(
    db: Session,
    row: db_models.Model,
    status: str,
    latency_ms: float,
    error_message: str | None,
    ctx: AuthContext,
    request: Request,
    input_summary: dict | None = None,
) -> None:
    db.add(
        db_models.InferenceLog(
            model_id=row.id,
            status=status,
            latency_ms=latency_ms,
            error_message=error_message,
            api_key_id=ctx.key_id,
            client_ip=client_ip(request),
            request_id=get_request_id(request),
            task_type=row.task_type,
            model_version=row.version,
            input_summary=input_summary,
        )
    )
    if ctx.key_id:
        db.query(db_models.ApiKey).filter(db_models.ApiKey.id == ctx.key_id).update(
            {db_models.ApiKey.request_count: db_models.ApiKey.request_count + 1},
            synchronize_session=False,
        )
    db.commit()
