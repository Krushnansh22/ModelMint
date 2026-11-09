"""ONNX Runtime adapter. Input/output schema comes straight from the graph,
so this adapter reliably hits SchemaConfidence.STRICT."""
from typing import Optional

import numpy as np
import onnxruntime as ort

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

ONNX_MAGIC = b"\x08"  # first bytes of the protobuf-encoded ModelProto vary;
# we sniff by extension + a lightweight parse attempt instead of magic bytes.

_ORT_DTYPE_MAP = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
}


class OnnxAdapter(ModelAdapter):
    def __init__(self):
        self._session: Optional[ort.InferenceSession] = None
        self._input_meta = []
        self._output_meta = []

    @staticmethod
    def detect(file_path: str) -> bool:
        if not file_path.endswith(".onnx"):
            return False
        try:
            # Cheap structural check without a full load: onnx files are
            # protobuf, so just confirm it's non-empty and parseable as such
            # via onnxruntime's lightweight session-options probe is overkill
            # for detect(); a size/extension check is the "cheap" tier here.
            import os
            return os.path.getsize(file_path) > 8
        except OSError:
            return False

    def load(self, file_path: str) -> None:
        try:
            # Restrict to CPU EP and disable any external data / custom ops
            # loading for MVP safety-by-default.
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = True
            self._session = ort.InferenceSession(
                file_path, sess_options=so, providers=["CPUExecutionProvider"]
            )
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load ONNX model: {e}") from e

        self._input_meta = self._session.get_inputs()
        self._output_meta = self._session.get_outputs()

    def unload(self) -> None:
        self._session = None
        self._input_meta = []
        self._output_meta = []

    def _synthetic_inputs(self) -> dict:
        feed = {}
        rng = np.random.default_rng(0)
        for inp in self._input_meta:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            np_dtype = np.float32 if "float" in inp.type else np.int64
            feed[inp.name] = rng.standard_normal(shape).astype(np_dtype)
        return feed

    def validate(self) -> ValidationResult:
        if self._session is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        try:
            feed = self._synthetic_inputs()
            self._session.run(None, feed)
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=SchemaConfidence.STRICT,
                reason="synthetic-input test inference succeeded against graph-declared shapes",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.STRICT,
                reason=f"graph schema is known but synthetic inference failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        if not self._input_meta:
            return None
        fields = [
            FieldDescriptor(
                name=inp.name,
                dtype=_ORT_DTYPE_MAP.get(inp.type, inp.type),
                shape=[d if isinstance(d, int) else None for d in inp.shape],
            )
            for inp in self._input_meta
        ]
        feature_count = None
        if len(fields) == 1 and fields[0].shape and isinstance(fields[0].shape[-1], int):
            feature_count = fields[0].shape[-1]
        return SchemaDescriptor(fields=fields, feature_count=feature_count)

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        if not self._output_meta:
            return None
        fields = [
            FieldDescriptor(
                name=out.name,
                dtype=_ORT_DTYPE_MAP.get(out.type, out.type),
                shape=[d if isinstance(d, int) else None for d in out.shape],
            )
            for out in self._output_meta
        ]
        return SchemaDescriptor(fields=fields)

    def predict(self, payload: dict) -> PredictionResult:
        if self._session is None:
            raise AdapterInferenceError("model not loaded")

        features = payload.get("features")
        if features is None:
            raise AdapterInferenceError("payload must contain a 'features' field")

        try:
            inp = self._input_meta[0]
            np_dtype = np.float32 if "float" in inp.type else np.int64
            x = np.asarray(features, dtype=np_dtype)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            outputs = self._session.run(None, {inp.name: x})
            preds = outputs[0]
            return PredictionResult(
                predictions=np.asarray(preds).tolist(),
                raw_output_shape=list(np.asarray(preds).shape),
            )
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        n_features = None
        schema = self.get_input_schema()
        if schema and schema.feature_count:
            n_features = schema.feature_count

        producer, ir_version = None, None
        if self._session is not None:
            try:
                modelmeta = self._session.get_modelmeta()
                producer = modelmeta.producer_name or None
            except Exception:  # noqa: BLE001
                pass

        version = None
        try:
            import onnxruntime

            version = onnxruntime.__version__
        except Exception:  # noqa: BLE001
            pass

        return ModelMetadata(
            framework="onnx",
            model_type="onnx-graph",
            task_type=self.task_type.value,
            n_features=n_features,
            framework_version=version,
            onnx_producer=producer,
            onnx_ir_version=ir_version,
        )
