"""
Upload -> detect -> (sandboxed) load -> validate pipeline.

Runs synchronously inside the upload request for MVP simplicity (the build
guide doesn't call for a background task queue, and sandboxed loads are
bounded by DEFAULT_TIMEOUT_SECONDS, so this is an acceptable blocking cost
for an upload endpoint). Writes final status back onto the Model row.
"""
from __future__ import annotations

import os
import time

from sqlalchemy.orm import Session

from app.database import models as db_models
from app.models.adapters.base import AdapterDetectionError, AdapterLoadError
from app.models.adapters.registry import detect_adapter, framework_name, is_sandboxed
from app.models.schemas import ValidationStatus, SchemaConfidence, ModelStatus
from app.services.sandbox import run_sandboxed_load_and_validate


def analyze_and_finalize(db: Session, model_row: db_models.Model) -> db_models.Model:
    file_path = model_row.file_path
    model_row.status = ModelStatus.ANALYZING.value
    db.commit()

    # --- detect() -----------------------------------------------------
    try:
        adapter_cls = detect_adapter(file_path)
    except AdapterDetectionError as e:
        model_row.status = ModelStatus.FAILED.value
        model_row.failure_reason = str(e)
        db.commit()
        return model_row

    framework = framework_name(adapter_cls)
    model_row.framework = framework

    # --- load() + validate() ------------------------------------------
    t0 = time.perf_counter()
    if is_sandboxed(framework):
        sandbox_result = run_sandboxed_load_and_validate(framework, file_path)
        if not sandbox_result.ok:
            model_row.status = ModelStatus.FAILED.value
            model_row.failure_reason = _sanitize_reason(sandbox_result.error or "load failed")
            db.commit()
            return model_row
        validation_status = sandbox_result.validation_status
        schema_confidence = sandbox_result.schema_confidence
        validation_reason = sandbox_result.validation_reason
        load_time_ms = sandbox_result.load_time_ms
        # Re-load in-process (outside the sandbox) purely to extract schema
        # + metadata for the registry row. The sandbox process already
        # proved the pickle is well-behaved under isolation; this second,
        # in-process load is for read-only introspection, never for serving
        # a request directly from this code path.
        adapter = adapter_cls()
        try:
            adapter.load(file_path)
        except AdapterLoadError as e:
            model_row.status = ModelStatus.FAILED.value
            model_row.failure_reason = f"passed sandbox but failed introspection reload: {e}"
            db.commit()
            return model_row
    else:
        adapter = adapter_cls()
        try:
            adapter.load(file_path)
        except AdapterLoadError as e:
            model_row.status = ModelStatus.FAILED.value
            model_row.failure_reason = str(e)
            db.commit()
            return model_row
        validation = adapter.validate()
        validation_status = validation.status.value
        schema_confidence = validation.schema_confidence.value
        validation_reason = validation.reason
        load_time_ms = (time.perf_counter() - t0) * 1000

    input_schema = adapter.get_input_schema()
    output_schema = adapter.get_output_schema()
    metadata = adapter.get_metadata()
    adapter.unload()

    model_row.model_type = metadata.model_type
    model_row.input_schema = input_schema.model_dump() if input_schema else None
    model_row.output_schema = output_schema.model_dump() if output_schema else None
    model_row.schema_confidence = schema_confidence
    model_row.load_time_ms = load_time_ms
    model_row.metadata_json = metadata.model_dump(exclude_none=True)
    model_row.failure_reason = validation_reason

    # --- final status decision (section 2.3 three-tier logic) ---------
    if validation_status == ValidationStatus.FAIL.value:
        model_row.status = ModelStatus.FAILED.value
    elif schema_confidence == SchemaConfidence.UNKNOWN.value:
        model_row.status = ModelStatus.READY_UNVALIDATED.value
    else:
        model_row.status = ModelStatus.READY.value

    db.commit()
    db.refresh(model_row)
    return model_row


def _sanitize_reason(raw: str) -> str:
    """Strip anything that looks like a sandbox-internal traceback path
    before it reaches a client-facing field. We keep the first line only."""
    return raw.splitlines()[0][:500] if raw else "unknown load failure"
