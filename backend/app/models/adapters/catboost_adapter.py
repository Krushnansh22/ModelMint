"""CatBoost adapter (.cbm artifacts)."""
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


class CatBoostAdapter(ModelAdapter):
    task_type = TaskType.TABULAR

    def __init__(self) -> None:
        self._model = None
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list[str]] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith(".cbm")

    def load(self, file_path: str) -> None:
        try:
            from catboost import CatBoost  # lazy: optional dependency
        except ImportError as e:
            raise AdapterLoadError(
                "CatBoost artifact detected but the 'catboost' package is not installed. "
                "Install backend/requirements-formats.txt."
            ) from e

        try:
            model = CatBoost.load_model(file_path)
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load CatBoost model: {e}") from e

        self._model = model
        names = getattr(model, "feature_names_", None)
        self._feature_names = list(names) if names else None
        n = getattr(model, "n_features_in_", None)
        if n is None and self._feature_names is not None:
            n = len(self._feature_names)
        self._n_features = n or None

    def unload(self) -> None:
        self._model = None
        self._n_features = None
        self._feature_names = None

    def _synthetic_input(self) -> np.ndarray:
        n = self._n_features or 4
        return np.random.default_rng(0).standard_normal((1, n)).astype(np.float32)

    def validate(self) -> ValidationResult:
        if self._model is None:
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
            self._model.predict(self._synthetic_input())
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
                fields=[FieldDescriptor(name=n, dtype="float32", shape=[1]) for n in self._feature_names],
                feature_count=len(self._feature_names),
            )
        if self._n_features:
            return SchemaDescriptor(fields=[], feature_count=self._n_features)
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._model is None:
            raise AdapterInferenceError("model not loaded")
        features = payload.get("features")
        if features is None:
            raise AdapterInferenceError("payload must contain a 'features' field")
        try:
            x = np.asarray(features, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if self._n_features is not None and x.shape[1] != self._n_features:
                raise AdapterInferenceError(
                    f"expected {self._n_features} features, got {x.shape[1]}"
                )
            preds = np.asarray(self._model.predict(x))
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
            import catboost

            version = catboost.__version__
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="catboost",
            model_type="CatBoost",
            task_type=self.task_type.value,
            n_features=self._n_features,
            framework_version=version,
        )
