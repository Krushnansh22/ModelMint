"""
Option A from build guide section 5: single generic /predict route, with
server-side schema-aware validation looked up from the model's stored
SchemaDescriptor rather than per-model FastAPI/Pydantic route typing.

M3 makes validation task-aware: tabular models use `features`, text models
use `text`/`texts`, vision models use `image_base64`, and embeddings accept
text or features.

(Option B -- dynamic route registration + OpenAPI schema rebuild per model
-- is intentionally not implemented; noted in the build guide as a
fast-follow if per-model Swagger typing becomes necessary.)
"""
from __future__ import annotations

from typing import Any, Optional

from app.database import models as db_models
from app.models.adapters.base import AdapterInferenceError


def _task_type(model_row: db_models.Model) -> str:
    return (model_row.task_type or "tabular").lower()


def validate_payload_against_schema(model_row: db_models.Model, payload: dict[str, Any]) -> None:
    """Raises AdapterInferenceError (mapped to HTTP 400 by the route) if the
    payload is invalid for the model's task/confidence tier."""
    task = _task_type(model_row)

    if task == "text":
        _validate_text_payload(payload)
        return

    if task == "vision":
        image = payload.get("image_base64")
        if not isinstance(image, str) or not image.strip():
            raise AdapterInferenceError(
                "payload must contain a non-empty 'image_base64' string"
            )
        return

    if task == "embedding":
        if payload.get("text") is None and payload.get("features") is None:
            raise AdapterInferenceError("payload must contain 'text' or 'features'")
        return

    # --- tabular (original behavior, kept backward compatible) -----------
    confidence = model_row.schema_confidence
    if confidence == "UNKNOWN" or confidence is None:
        return  # best-effort, no pre-validation possible

    features = payload.get("features")
    if features is None:
        raise AdapterInferenceError("payload must contain a 'features' field")

    schema = model_row.input_schema or {}
    expected: Optional[int] = schema.get("feature_count")
    if expected is None:
        return

    length = _feature_length(features)
    if length != expected:
        raise AdapterInferenceError(
            f"expected {expected} features (schema_confidence={confidence}), got {length}"
        )


def _validate_text_payload(payload: dict[str, Any]) -> None:
    text = payload.get("text")
    prompt = payload.get("prompt")
    texts = payload.get("texts")
    if text is None and texts is None and prompt is None:
        raise AdapterInferenceError("payload must contain 'text', 'prompt', or 'texts'")
    if text is not None and not isinstance(text, str):
        raise AdapterInferenceError("'text' must be a string")
    if prompt is not None and not isinstance(prompt, str):
        raise AdapterInferenceError("'prompt' must be a string")
    if texts is not None:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise AdapterInferenceError("'texts' must be a list of strings")


def _feature_length(features: Any) -> int:
    if isinstance(features, list) and features and isinstance(features[0], list):
        return len(features[0])
    if isinstance(features, list):
        return len(features)
    return -1


def build_example_request(model_row: db_models.Model) -> dict:
    """Builds a task-aware example request for the Model Details page."""
    task = _task_type(model_row)

    if task == "text":
        return {"text": "The quick brown fox jumps over the lazy dog."}
    if task == "vision":
        return {"image_base64": "<base64-encoded image bytes>"}
    if task == "embedding":
        return {"text": "embed this sentence"}

    schema = model_row.input_schema or {}
    fields = schema.get("fields") or []
    if fields:
        example = [1.0 for _ in fields]
    elif schema.get("feature_count"):
        example = [1.0 for _ in range(schema["feature_count"])]
    else:
        example = []
    return {"features": example}
