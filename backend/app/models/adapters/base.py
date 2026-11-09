"""
ModelAdapter interface (build guide section 3) + the typed exception
hierarchy every adapter must raise instead of bare exceptions (section 9).
"""
from abc import ABC, abstractmethod
from typing import Optional

from app.models.schemas import (
    InputContract,
    ModelMetadata,
    PredictionResult,
    SchemaDescriptor,
    TaskType,
    ValidationResult,
)


class AdapterDetectionError(Exception):
    """File doesn't match any known adapter. -> HTTP 400"""


class AdapterLoadError(Exception):
    """Model failed to load: corrupt file, incompatible version, sandbox
    timeout, or the sandboxed worker crashed. -> HTTP 422"""


class AdapterValidationError(Exception):
    """Loaded, but synthetic-input test inference failed outright."""


class AdapterInferenceError(Exception):
    """Runtime prediction failure: shape mismatch, bad input. -> HTTP 400"""


class ModelAdapter(ABC):
    """One instance per loaded model. Instances are managed (and evicted) by
    the Model Manager (app/models/registry.py); adapters never manage their
    own lifecycle beyond load()/unload()."""

    # Task/contract defaults; adapters override these as needed. Keeping them
    # as plain class attributes means existing adapters stay source-compatible.
    task_type: TaskType = TaskType.TABULAR
    input_contract: InputContract = InputContract.FEATURES

    def build_example_payload(self) -> dict:
        """Task-aware example body for the Model Details page."""
        schema = self.get_input_schema()
        n = schema.feature_count if schema and schema.feature_count else 4
        return {"features": [1.0] * n}

    def predict_batch(self, payloads: list[dict]) -> list[PredictionResult]:
        """Default batch implementation: sequential per-item predict()."""
        return [self.predict(payload) for payload in payloads]

    @staticmethod
    @abstractmethod
    def detect(file_path: str) -> bool:
        """Cheap, fast check: can this adapter plausibly handle this file?
        Must NOT fully load the model -- magic bytes / extension / header
        sniff only."""
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Full load into memory. For pickle-based adapters this is called
        INSIDE the sandboxed worker process, never directly by a route
        handler. Must raise AdapterLoadError on failure."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Release memory / underlying session. Must be safe to call even if
        load() was never called (idempotent)."""
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Runs synthetic-input test inference against the loaded model."""
        raise NotImplementedError

    @abstractmethod
    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        """None is valid -- means UNKNOWN confidence. Caller must handle it."""
        raise NotImplementedError

    @abstractmethod
    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, payload: dict) -> PredictionResult:
        """Must be safe to call from a threadpool (sync, CPU-bound). Raises
        AdapterInferenceError, never a bare exception."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns only the fields applicable to this framework -- omit,
        don't null-fill, metrics that don't apply."""
        raise NotImplementedError
