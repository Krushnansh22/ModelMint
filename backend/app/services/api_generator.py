"""
Option A from build guide section 5: single generic /predict route, with
server-side schema-aware validation looked up from the model's stored
SchemaDescriptor rather than per-model FastAPI/Pydantic route typing.

(Option B -- dynamic route registration + OpenAPI schema rebuild per model
-- is intentionally not implemented; noted in the build guide as a
fast-follow if per-model Swagger typing becomes necessary.)
"""
from __future__ import annotations

from typing import Any, Optional

from app.database import models as db_models
from app.models.adapters.base import AdapterInferenceError


def validate_payload_against_schema(model_row: db_models.Model, payload: dict[str, Any]) -> None:
    """Raises AdapterInferenceError (mapped to HTTP 400 by the route) if the
    payload is invalid for the given confidence tier. STRICT/INFERRED check
    feature count; UNKNOWN accepts anything and defers entirely to the
    adapter's own best-effort predict()."""
    confidence = model_row.schema_confidence
    if confidence == "UNKNOWN" or confidence is None:
        return  # best-effort, no pre-validation possible

    features = payload.get("features")
    if features is None:
        raise AdapterInferenceError("payload must contain a 'features' field")

    expected: Optional[int] = None
    schema = model_row.input_schema or {}
    expected = schema.get("feature_count")

    if expected is None:
        return

    length = _feature_length(features)
    if length != expected:
        raise AdapterInferenceError(
            f"expected {expected} features (schema_confidence={confidence}), got {length}"
        )


def _feature_length(features: Any) -> int:
    if isinstance(features, list) and features and isinstance(features[0], list):
        return len(features[0])
    if isinstance(features, list):
        return len(features)
    return -1


def build_example_request(model_row: db_models.Model) -> dict:
    """Builds a typed example request for the Model Details page (which
    fully owns the "nicely typed" experience per Option A, since real
    Swagger only ever shows the generic shape)."""
    schema = model_row.input_schema or {}
    fields = schema.get("fields") or []
    if fields:
        example = [1.0 for _ in fields]
    elif schema.get("feature_count"):
        example = [1.0 for _ in range(schema["feature_count"])]
    else:
        example = "unknown schema -- send a raw JSON body, e.g. {\"features\": [...]}"
    return {"features": example}
