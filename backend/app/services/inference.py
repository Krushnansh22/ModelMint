"""Threadpool dispatch wrapper for predict() calls (build guide section 4
threading note): sklearn/ONNX predict() is blocking + CPU-bound, so it must
never run inline inside an `async def` route."""
from __future__ import annotations

import time

from starlette.concurrency import run_in_threadpool

from app.models.adapters.base import ModelAdapter, AdapterInferenceError
from app.models.schemas import PredictionResult


async def dispatch_predict(adapter: ModelAdapter, payload: dict) -> tuple[PredictionResult, float]:
    """Runs adapter.predict() in FastAPI/Starlette's threadpool. Returns
    (result, latency_ms). Re-raises AdapterInferenceError unchanged so the
    API layer can map it to the standard error contract."""
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(adapter.predict, payload)
    except AdapterInferenceError:
        raise
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms
