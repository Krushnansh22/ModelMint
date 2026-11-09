"""LightGBM adapter (text model files)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.models.adapters.base import AdapterInferenceError, AdapterLoadError, ModelAdapter
from app.models.schemas import (
    ModelMetadata,
    PredictionResult,
    SchemaConfidence,
    SchemaDescriptor,
    TaskType,
    ValidationResult,
    ValidationStatus,
)


class LightGBMAdapter(ModelAdapter):
    task_type = TaskType.TABULAR

    def __init__(self) -> None:
        self._booster = None
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list[str]] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith((".txt", ".lgbm", ".model"))

    def load(self, file_path: str) -> None:
        try:
            import lightgbm  # lazy: optional dependency
        except ImportError as e:
            raise AdapterLoadError(
                "LightGBM artifact detected but the 'lightgbm' package is not installed. "
                "Install backend/requirements-formats.txt."
            ) from e

        try:
            booster = lightgbm.Booster(model_file=file_path)
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load LightGBM model: {e}") from e

        self._booster = booster
        self._n_features = booster.num_feature() or None
        names = booster.feature_name()
        self._feature_names = list(names) if names else None

    def unload(self) -> None:
        self._booster = None
        self._n_features = None
        self._feature_names = None

    def _synthetic_input(self) -> np.ndarray:
        n = self._n_features or 4
        return np.random.default_rng(0).standard_normal((1, n)).astype(np.float64)

    def validate(self) -> ValidationResult:
        if self._booster is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        confidence = (
            SchemaConfidence.STRICT
            if self._feature_names
            else SchemaConfidence.INFERRED
            if self._n_features is not None
            else SchemaConfidence.UNKNOWN
        )
        try:
            self._booster.predict(self._synthetic_input())
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=confidence,
                reason="synthetic-input test inference succeeded",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=confidence,
                reason=f"synthetic inference failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        if self._feature_names:
            from app.models.schemas import FieldDescriptor

            return SchemaDescriptor(
                fields=[FieldDescriptor(name=n, dtype="float64", shape=[1]) for n in self._feature_names],
                feature_count=len(self._feature_names),
            )
        if self._n_features:
            return SchemaDescriptor(fields=[], feature_count=self._n_features)
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._booster is None:
            raise AdapterInferenceError("model not loaded")
        features = payload.get("features")
        if features is None:
            raise AdapterInferenceError("payload must contain a 'features' field")
        try:
            x = np.asarray(features, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if self._n_features is not None and x.shape[1] != self._n_features:
                raise AdapterInferenceError(
                    f"expected {self._n_features} features, got {x.shape[1]}"
                )
            preds = np.asarray(self._booster.predict(x))
            return PredictionResult(
                predictions=preds.tolist(),
                raw_output_shape=list(preds.shape),
            )
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        version = None
        try:
            import lightgbm

            version = lightgbm.__version__
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="lightgbm",
            model_type="Booster",
            task_type=self.task_type.value,
            n_features=self._n_features,
            framework_version=version,
        )
