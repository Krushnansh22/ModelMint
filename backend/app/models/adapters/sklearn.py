"""
sklearn / joblib-pickle adapter.

IMPORTANT: load() executes untrusted pickle bytes. This adapter is only ever
instantiated and load()-ed from inside the sandboxed subprocess dispatched by
app/services/sandbox.py -- never called directly from a route handler.
"""
import time
from typing import Optional

import joblib
import numpy as np

from app.models.adapters.base import (
    ModelAdapter,
    AdapterLoadError,
    AdapterInferenceError,
)
from app.models.schemas import (
    ValidationResult,
    ValidationStatus,
    SchemaConfidence,
    SchemaDescriptor,
    FieldDescriptor,
    ModelMetadata,
    PredictionResult,
)

class SklearnAdapter(ModelAdapter):
    def __init__(self):
        self._estimator = None
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list[str]] = None
        self._n_classes: Optional[int] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        # Cheap tier: extension check only. We deliberately don't try to
        # sniff pickle opcode bytes here -- joblib.dump() output varies
        # (raw pickle stream, zlib-compressed, or joblib's own
        # NumpyArrayWrapper-prefixed format for large arrays), so a single
        # magic-byte check would false-negative on legitimate files. Full
        # format validation happens in load(), inside the sandbox.
        return file_path.endswith(".pkl") or file_path.endswith(".joblib")

    def load(self, file_path: str) -> None:
        try:
            # joblib.load() transparently handles plain pickle streams,
            # joblib's own numpy-array-wrapped format, and zlib/gzip
            # compressed variants -- pickle.load() alone only handles the
            # first of those, which silently breaks on real-world sklearn
            # pickles saved via joblib.dump() with large numpy arrays.
            estimator = joblib.load(file_path)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, re-raised typed
            raise AdapterLoadError(f"failed to unpickle model: {e}") from e

        if not hasattr(estimator, "predict"):
            raise AdapterLoadError(
                "unpickled object has no predict() method -- not a recognizable "
                "sklearn-style estimator"
            )

        self._estimator = estimator
        self._n_features = getattr(estimator, "n_features_in_", None)
        names = getattr(estimator, "feature_names_in_", None)
        self._feature_names = list(names) if names is not None else None
        classes = getattr(estimator, "classes_", None)
        self._n_classes = len(classes) if classes is not None else None

    def unload(self) -> None:
        self._estimator = None
        self._n_features = None
        self._feature_names = None
        self._n_classes = None

    def _synthetic_input(self) -> np.ndarray:
        n = self._n_features or 4
        rng = np.random.default_rng(0)
        return rng.standard_normal((1, n)).astype(np.float64)

    def validate(self) -> ValidationResult:
        if self._estimator is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )

        confidence = (
            SchemaConfidence.STRICT
            if self._feature_names is not None
            else SchemaConfidence.INFERRED
            if self._n_features is not None
            else SchemaConfidence.UNKNOWN
        )

        try:
            x = self._synthetic_input()
            self._estimator.predict(x)
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=confidence,
                reason="synthetic-input test inference succeeded",
            )
        except Exception as e:  # noqa: BLE001
            if confidence != SchemaConfidence.UNKNOWN:
                # We knew the shape and it still failed -- partial credit,
                # mark READY_UNVALIDATED territory rather than outright FAIL.
                return ValidationResult(
                    status=ValidationStatus.PARTIAL,
                    schema_confidence=confidence,
                    reason=f"synthetic inference failed despite known feature count: {e}",
                )
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason=(
                    "could not determine input shape and synthetic inference "
                    f"failed ({e}) -- accepting as READY_UNVALIDATED, first "
                    "real request will attempt best-effort inference"
                ),
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        if self._feature_names is not None:
            return SchemaDescriptor(
                fields=[FieldDescriptor(name=n, dtype="float64", shape=[1]) for n in self._feature_names],
                feature_count=len(self._feature_names),
            )
        if self._n_features is not None:
            return SchemaDescriptor(fields=[], feature_count=self._n_features)
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        if self._n_classes is not None:
            return SchemaDescriptor(fields=[FieldDescriptor(name="class", dtype="int64", shape=[1])])
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._estimator is None:
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
            t0 = time.perf_counter()
            preds = self._estimator.predict(x)
            _ = time.perf_counter() - t0
            return PredictionResult(
                predictions=np.atleast_1d(preds).tolist(),
                raw_output_shape=list(np.atleast_1d(preds).shape),
            )
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        params = None
        if self._estimator is not None and hasattr(self._estimator, "get_params"):
            try:
                raw = self._estimator.get_params()
                params = {k: v for k, v in raw.items() if _is_jsonable(v)}
            except Exception:  # noqa: BLE001
                params = None

        version = None
        try:
            import sklearn

            version = sklearn.__version__
        except Exception:  # noqa: BLE001
            pass

        return ModelMetadata(
            framework="sklearn",
            model_type=type(self._estimator).__name__ if self._estimator else None,
            task_type=self.task_type.value,
            n_features=self._n_features,
            n_classes=self._n_classes,
            framework_version=version,
            estimator_params=params,
        )


def _is_jsonable(value) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))
