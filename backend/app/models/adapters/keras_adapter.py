"""Keras adapter (`.keras` / legacy `.h5` SavedModel-style artifacts).

Keras models are untrusted native code, so this adapter is served through a
persistent sandboxed worker like sklearn/TorchScript. We never enable
`custom_objects` or `safe_mode=False`: a bundle that needs arbitrary Python
classes to load is not servable here.

Task detection is shape-driven: a 2-D input is tabular (`features`), a 4-D
input is vision (`image_base64`). Both channels-first and channels-last
layouts are supported.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np

from app.models.adapters.base import AdapterInferenceError, AdapterLoadError, ModelAdapter
from app.models.schemas import (
    FieldDescriptor,
    InputContract,
    ModelMetadata,
    PredictionResult,
    SchemaConfidence,
    SchemaDescriptor,
    TaskType,
    ValidationResult,
    ValidationStatus,
)


def _import_keras():
    """Return a Keras module from any of the common providers."""
    try:
        import keras  # type: ignore

        return keras
    except ImportError:
        pass
    try:
        import tf_keras  # type: ignore

        return tf_keras
    except ImportError:
        pass
    try:
        from tensorflow import keras  # type: ignore

        return keras
    except ImportError as e:  # pragma: no cover - exercised only without TF
        raise AdapterLoadError(
            "Keras artifact detected but neither 'keras', 'tf-keras', nor "
            "'tensorflow' is installed. Install backend/requirements-formats.txt."
        ) from e


class KerasAdapter(ModelAdapter):
    task_type = TaskType.TABULAR
    input_contract = InputContract.FEATURES

    def __init__(self) -> None:
        self._model = None
        self._input_shape: Optional[tuple[int, ...]] = None
        self._channels_first = False
        self._n_features: Optional[int] = None
        self._n_classes: Optional[int] = None
        self._keras = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith((".keras", ".h5"))

    def load(self, file_path: str) -> None:
        keras = _import_keras()
        try:
            model = keras.models.load_model(file_path, compile=False)
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load Keras model: {e}") from e

        self._keras = keras
        self._model = model
        self._input_shape = _normalize_input_shape(getattr(model, "input_shape", None))
        self._channels_first = _is_channels_first(self._input_shape)
        self._configure_task()

    def _configure_task(self) -> None:
        shape = self._input_shape
        if not shape:
            return
        if len(shape) == 1:
            self._n_features = shape[0]
            self.task_type = TaskType.TABULAR
            self.input_contract = InputContract.FEATURES
        elif len(shape) == 3:
            # (H, W, C) or (C, H, W)
            self.task_type = TaskType.VISION
            self.input_contract = InputContract.IMAGE
            self._n_features = None
        else:
            self._n_features = int(np.prod(shape)) if shape else None
            self.task_type = TaskType.TABULAR
            self.input_contract = InputContract.FEATURES

        output_shape = _normalize_input_shape(getattr(self._model, "output_shape", None))
        if output_shape and output_shape[-1] and output_shape[-1] > 1:
            self._n_classes = output_shape[-1]

    def unload(self) -> None:
        self._model = None
        self._keras = None
        self._input_shape = None
        self._n_features = None
        self._n_classes = None

    def _zeros(self) -> np.ndarray:
        shape = self._input_shape or (4,)
        return np.zeros((1, *shape), dtype=np.float32)

    def validate(self) -> ValidationResult:
        if self._model is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        try:
            self._model.predict(self._zeros(), verbose=0)
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=(
                    SchemaConfidence.INFERRED if self._input_shape else SchemaConfidence.UNKNOWN
                ),
                reason="synthetic-input test inference succeeded",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason=f"synthetic inference failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        if self._n_features:
            return SchemaDescriptor(fields=[], feature_count=self._n_features)
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        if self._n_classes:
            return SchemaDescriptor(
                fields=[
                    FieldDescriptor(name="class", dtype="float32", shape=[self._n_classes])
                ]
            )
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._model is None:
            raise AdapterInferenceError("model not loaded")
        try:
            x = self._prepare_input(payload)
            output = np.asarray(self._model.predict(x, verbose=0))
            return PredictionResult(predictions=output.tolist(), raw_output_shape=list(output.shape))
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def _prepare_input(self, payload: dict) -> np.ndarray:
        shape = self._input_shape
        if self.task_type == TaskType.VISION:
            image = payload.get("image_base64")
            if not isinstance(image, str) or not image.strip():
                raise AdapterInferenceError("payload must contain a non-empty 'image_base64' string")
            array = _decode_image(image, shape, self._channels_first)
            return array.astype(np.float32)

        features = payload.get("features")
        if features is None:
            raise AdapterInferenceError("payload must contain a 'features' field")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if shape:
            expected = int(np.prod(shape))
            if x.size != expected:
                raise AdapterInferenceError(
                    f"expected {expected} values for input shape {list(shape)}, got {x.size}"
                )
            x = x.reshape((1, *shape))
        return x

    def get_metadata(self) -> ModelMetadata:
        version = None
        try:
            version = getattr(self._keras, "__version__", None)
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="keras",
            model_type=type(self._model).__name__ if self._model is not None else None,
            task_type=self.task_type.value,
            input_contract=self.input_contract.value,
            n_features=self._n_features,
            n_classes=self._n_classes,
            framework_version=version,
            extra={
                "input_shape": list(self._input_shape) if self._input_shape else None,
                "channels_first": self._channels_first,
            },
        )


def _normalize_input_shape(raw) -> Optional[tuple[int, ...]]:
    """Keras returns a tuple, a list of tuples, or None. Drop the batch dim."""
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    try:
        shape = tuple(int(d) for d in raw if d is not None)
    except (TypeError, ValueError):
        return None
    return shape or None


def _is_channels_first(shape: Optional[tuple[int, ...]]) -> bool:
    if not shape or len(shape) != 3:
        return False
    # (C, H, W) is channels-first when the first dim is a small channel count
    # and the remaining dims are spatial.
    return shape[0] in (1, 3, 4) and shape[1] > 4 and shape[2] > 4


def _decode_image(image_b64: str, shape: Optional[tuple[int, ...]], channels_first: bool) -> np.ndarray:
    try:
        from PIL import Image  # lazy: optional, shipped with TF/transformers
    except ImportError as e:  # pragma: no cover - exercised only without Pillow
        raise AdapterInferenceError(
            "image payloads require Pillow; install backend/requirements-formats.txt"
        ) from e

    try:
        raw = base64.b64decode(image_b64, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise AdapterInferenceError(f"could not decode 'image_base64': {e}") from e

    if shape:
        height, width = _spatial_dims(shape, channels_first)
        if height and width:
            image = image.resize((width, height))
    array = np.asarray(image, dtype=np.float32)
    if channels_first:
        array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _spatial_dims(shape: tuple[int, ...], channels_first: bool) -> tuple[Optional[int], Optional[int]]:
    if len(shape) != 3:
        return None, None
    if channels_first:
        return shape[1], shape[2]
    return shape[0], shape[1]
