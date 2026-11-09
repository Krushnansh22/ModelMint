"""TensorFlow Lite adapter (`.tflite` flatbuffer models).

Served through the persistent sandboxed worker: TFLite can run native and
custom operators, so the interpreter never lives in the API process.

The interpreter's input details give us the exact shape and dtype, which
makes schema inference strict. A 2-D input is tabular; a 4-D input is vision
(`image_base64`).
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


def _interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter, "tflite_runtime"
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # type: ignore

        return Interpreter, "tensorflow"
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

        return Interpreter, "tensorflow"
    except ImportError as e:  # pragma: no cover - exercised only without TFLite
        raise AdapterLoadError(
            "TFLite artifact detected but neither 'tflite-runtime' nor "
            "'tensorflow' is installed. Install backend/requirements-formats.txt."
        ) from e


class TFLiteAdapter(ModelAdapter):
    task_type = TaskType.TABULAR
    input_contract = InputContract.FEATURES

    def __init__(self) -> None:
        self._interpreter = None
        self._provider = None
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._input_shape: Optional[tuple[int, ...]] = None
        self._input_dtype = np.float32
        self._n_features: Optional[int] = None
        self._channels_first = False

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith(".tflite")

    def load(self, file_path: str) -> None:
        Interpreter, provider = _interpreter_class()
        try:
            interpreter = Interpreter(model_path=file_path)
            interpreter.allocate_tensors()
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load TFLite model: {e}") from e

        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if not inputs or not outputs:
            raise AdapterLoadError("TFLite model has no input or output tensors")

        detail = inputs[0]
        self._interpreter = interpreter
        self._provider = provider
        self._input_index = int(detail["index"])
        self._output_index = int(outputs[0]["index"])
        self._input_shape = tuple(int(d) if int(d) > 0 else 1 for d in detail["shape"])
        self._input_dtype = np.dtype(detail["dtype"])
        self._channels_first = _is_channels_first(self._input_shape)
        self._configure_task()

    def _configure_task(self) -> None:
        shape = self._input_shape
        if not shape:
            return
        # Drop the batch dimension.
        dims = shape[1:] if len(shape) > 1 else shape
        if len(dims) == 1:
            self._n_features = dims[0]
            self.task_type = TaskType.TABULAR
            self.input_contract = InputContract.FEATURES
        elif len(dims) == 3:
            self.task_type = TaskType.VISION
            self.input_contract = InputContract.IMAGE
            self._n_features = None
        else:
            self._n_features = int(np.prod(dims)) if dims else None
            self.task_type = TaskType.TABULAR
            self.input_contract = InputContract.FEATURES

    def unload(self) -> None:
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._input_shape = None
        self._n_features = None

    def _invoke(self, array: np.ndarray):
        self._interpreter.set_tensor(self._input_index, array.astype(self._input_dtype))
        self._interpreter.invoke()
        return np.asarray(self._interpreter.get_tensor(self._output_index))

    def validate(self) -> ValidationResult:
        if self._interpreter is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        try:
            self._invoke(np.zeros(self._input_shape, dtype=self._input_dtype))
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=SchemaConfidence.STRICT,
                reason="synthetic-input test inference succeeded",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.INFERRED,
                reason=f"synthetic inference failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        if self._n_features:
            return SchemaDescriptor(
                fields=[], feature_count=self._n_features
            )
        if self.task_type == TaskType.VISION and self._input_shape:
            dims = self._input_shape[1:]
            return SchemaDescriptor(
                fields=[
                    FieldDescriptor(name="image", dtype=str(self._input_dtype), shape=list(dims))
                ]
            )
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._interpreter is None:
            raise AdapterInferenceError("model not loaded")
        try:
            x = self._prepare_input(payload)
            output = self._invoke(x)
            return PredictionResult(predictions=output.tolist(), raw_output_shape=list(output.shape))
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def _prepare_input(self, payload: dict) -> np.ndarray:
        if self.task_type == TaskType.VISION:
            image = payload.get("image_base64")
            if not isinstance(image, str) or not image.strip():
                raise AdapterInferenceError("payload must contain a non-empty 'image_base64' string")
            return _decode_image(image, self._input_shape, self._channels_first)

        features = payload.get("features")
        if features is None:
            raise AdapterInferenceError("payload must contain a 'features' field")
        x = np.asarray(features, dtype=self._input_dtype)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        expected = int(np.prod(self._input_shape[1:])) if self._input_shape else x.size
        if x.size != expected:
            raise AdapterInferenceError(
                f"expected {expected} values for input shape {list(self._input_shape or [])}, got {x.size}"
            )
        return x.reshape(self._input_shape)

    def get_metadata(self) -> ModelMetadata:
        version = None
        try:
            if self._provider == "tflite_runtime":
                import tflite_runtime  # type: ignore

                version = getattr(tflite_runtime, "__version__", None)
            else:
                import tensorflow  # type: ignore

                version = getattr(tensorflow, "__version__", None)
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="tflite",
            model_type="TFLite Interpreter",
            task_type=self.task_type.value,
            input_contract=self.input_contract.value,
            n_features=self._n_features,
            framework_version=version,
            extra={
                "input_shape": list(self._input_shape) if self._input_shape else None,
                "input_dtype": str(self._input_dtype),
                "channels_first": self._channels_first,
                "provider": self._provider,
            },
        )


def _is_channels_first(shape: Optional[tuple[int, ...]]) -> bool:
    if not shape or len(shape) != 4:
        return False
    dims = shape[1:]
    return dims[0] in (1, 3, 4) and dims[1] > 4 and dims[2] > 4


def _decode_image(image_b64: str, shape: Optional[tuple[int, ...]], channels_first: bool) -> np.ndarray:
    try:
        from PIL import Image  # lazy: optional
    except ImportError as e:  # pragma: no cover
        raise AdapterInferenceError(
            "image payloads require Pillow; install backend/requirements-formats.txt"
        ) from e

    try:
        raw = base64.b64decode(image_b64, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise AdapterInferenceError(f"could not decode 'image_base64': {e}") from e

    if shape and len(shape) == 4:
        height, width = (shape[2], shape[3]) if channels_first else (shape[1], shape[2])
        image = image.resize((width, height))
    array = np.asarray(image, dtype=np.float32)
    if channels_first:
        array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)
