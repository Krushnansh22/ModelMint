"""Threadpool dispatch for sync, CPU-bound adapter inference."""
from __future__ import annotations

import time

from starlette.concurrency import run_in_threadpool

from app.models.adapters.base import ModelAdapter
from app.models.schemas import PredictionResult


async def dispatch_predict(adapter: ModelAdapter, payload: dict) -> tuple[PredictionResult, float]:
    t0 = time.perf_counter()
    result = await run_in_threadpool(adapter.predict, payload)
    return result, (time.perf_counter() - t0) * 1000


async def dispatch_predict_batch(
    adapter: ModelAdapter, payloads: list[dict]
) -> tuple[list[PredictionResult], float]:
    t0 = time.perf_counter()
    results = await run_in_threadpool(adapter.predict_batch, payloads)
    return results, (time.perf_counter() - t0) * 1000
