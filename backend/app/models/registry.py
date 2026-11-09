"""
Model Manager: registry lookups + in-memory adapter lifecycle + LRU eviction
(build guide section 2.2 and section 4's "Model Manager" box).

This is the single place that holds resident (loaded) adapter instances. It
does not perform upload-time analysis (that is services/analyzer.py). What
THIS module does is:

  1. Keep a bounded, LRU-evicted cache of live ModelAdapter instances so the
     inference route never reloads a model per-request.
  2. Re-load a model on demand when a request comes in for a model that's
     been evicted or was never resident. Sandboxed frameworks load into a
     persistent worker (`SandboxedAdapterProxy`), not this process.
  3. Evict least-recently-used adapters once the resident count exceeds
     MAX_RESIDENT_MODELS, unloading/killing their workers.

Cold-start latency on re-load-after-eviction is intentionally NOT hidden --
callers can inspect `was_cold_start` on the result if they want to log it.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.config import get_settings
from app.models.adapters.base import ModelAdapter, AdapterLoadError
from app.models.adapters.registry import adapter_for_framework, is_sandboxed
from app.services.sandboxed_adapter import SandboxedAdapterProxy

MAX_RESIDENT_MODELS = get_settings().max_resident_models


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
        """Returns (adapter, was_cold_start).

        Sandboxed frameworks (sklearn pickle, and later torch/Keras/HF) are
        served by a persistent worker process via `SandboxedAdapterProxy` --
        the untrusted model object never enters the API process. Non-sandboxed
        formats (ONNX, tree-model JSON) load in-process.
        """
        with self._lock:
            resident = self._resident.get(model_id)
            if resident is not None:
                resident.last_used = time.time()
                self._resident.move_to_end(model_id)
                return resident.adapter, False

        # Not resident -- load outside the lock (loading can be slow; don't
        # block other requests hitting already-resident models).
        adapter_cls = adapter_for_framework(framework)
        if is_sandboxed(framework):
            adapter = SandboxedAdapterProxy(framework)
            adapter.load(file_path)
        else:
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
