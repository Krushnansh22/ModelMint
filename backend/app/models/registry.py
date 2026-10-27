"""
Model Manager: registry lookups + in-memory adapter lifecycle + LRU eviction
(build guide section 2.2 and section 4's "Model Manager" box).

This is the single place that holds resident (loaded) adapter instances. It
does NOT talk to the sandbox directly for the *first* load of a pickle-based
model -- that dispatch happens in services/analyzer.py during upload
analysis. What THIS module does is:

  1. Keep a bounded, LRU-evicted cache of live ModelAdapter instances so the
     inference route never reloads a model per-request.
  2. Re-load a model on demand (via the same sandboxed-if-needed path) when
     a request comes in for a model that's been evicted or never resident.
  3. Evict least-recently-used adapters once the resident count exceeds
     MAX_RESIDENT_MODELS.

Cold-start latency on re-load-after-eviction is intentionally NOT hidden --
callers can inspect `was_cold_start` on the result if they want to log it.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.models.adapters.base import ModelAdapter, AdapterLoadError
from app.models.adapters.registry import adapter_for_framework, is_sandboxed
from app.services.sandbox import run_sandboxed_load_and_validate

MAX_RESIDENT_MODELS = 5


@dataclass
class ResidentModel:
    adapter: ModelAdapter
    last_used: float


class ModelManager:
    """Thread-safe LRU cache of resident adapter instances, keyed by model id (str)."""

    def __init__(self, max_resident: int = MAX_RESIDENT_MODELS):
        self._max_resident = max_resident
        self._resident: "OrderedDict[str, ResidentModel]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_load(self, model_id: str, framework: str, file_path: str) -> tuple[ModelAdapter, bool]:
        """Returns (adapter, was_cold_start). Loads in-process for ONNX;
        for sklearn/pickle-based frameworks, re-validates via the sandbox
        first (cheap relative to the risk of skipping it) then loads
        in-process for actual serving, matching the pattern used at upload
        time in services/analyzer.py."""
        with self._lock:
            resident = self._resident.get(model_id)
            if resident is not None:
                resident.last_used = time.time()
                self._resident.move_to_end(model_id)
                return resident.adapter, False

        # Not resident -- load outside the lock (loading can be slow; don't
        # block other requests hitting already-resident models).
        if is_sandboxed(framework):
            sandbox_result = run_sandboxed_load_and_validate(framework, file_path)
            if not sandbox_result.ok:
                raise AdapterLoadError(f"re-load failed sandbox re-validation: {sandbox_result.error}")

        adapter_cls = adapter_for_framework(framework)
        adapter = adapter_cls()
        adapter.load(file_path)

        with self._lock:
            self._resident[model_id] = ResidentModel(adapter=adapter, last_used=time.time())
            self._resident.move_to_end(model_id)
            self._evict_if_needed_locked()

        return adapter, True

    def evict(self, model_id: str) -> None:
        with self._lock:
            resident = self._resident.pop(model_id, None)
        if resident is not None:
            resident.adapter.unload()

    def _evict_if_needed_locked(self) -> None:
        while len(self._resident) > self._max_resident:
            evicted_id, evicted = self._resident.popitem(last=False)  # LRU = front
            evicted.adapter.unload()

    def resident_ids(self) -> list[str]:
        with self._lock:
            return list(self._resident.keys())


# Process-wide singleton -- the FastAPI app has a single Model Manager
# instance shared across all requests (via app.state, wired in main.py).
model_manager = ModelManager()
