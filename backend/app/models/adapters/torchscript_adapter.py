"""TorchScript adapter (traced/scripted PyTorch models).

A raw `state_dict`/`.safetensors` file is intentionally NOT handled here: it
has no architecture and cannot be served without user code. Export to
TorchScript (`torch.jit.trace` / `torch.jit.script`) or use an HF bundle.

This adapter is sandboxed: TorchScript can execute arbitrary operators, so
`SandboxedAdapterProxy` owns the actual model process.
"""
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

_KNOWN_INPUT_ATTRS = ("in_features", "n_features", "num_features", "input_dim")


class TorchScriptAdapter(ModelAdapter):
    task_type = TaskType.TABULAR

    def __init__(self) -> None:
        self._model = None
        self._n_features: Optional[int] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith((".pt", ".pth", ".torchscript"))

    def load(self, file_path: str) -> None:
        try:
            import torch  # lazy: optional dependency
        except ImportError as e:
            raise AdapterLoadError(
                "PyTorch artifact detected but the 'torch' package is not installed. "
                "Install backend/requirements-formats.txt."
            ) from e

        try:
            model = torch.jit.load(file_path, map_location="cpu")
            model.eval()
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(
                f"failed to load TorchScript model (a raw state_dict is not servable): {e}"
            ) from e

        self._model = model
        for attr in _KNOWN_INPUT_ATTRS:
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                self._n_features = value
                break

    def unload(self) -> None:
        self._model = None
        self._n_features = None

    def _run(self, n_features: int):
        import torch

        tensor = torch.zeros((1, n_features), dtype=torch.float32)
        with torch.no_grad():
            return self._model(tensor)

    def validate(self) -> ValidationResult:
        if self._model is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )

        candidates = [self._n_features] if self._n_features else [4, 1, 3]
        last_error = None
        for n in candidates:
            try:
                self._run(int(n))
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    schema_confidence=(
                        SchemaConfidence.INFERRED
                        if self._n_features
                        else SchemaConfidence.UNKNOWN
                    ),
                    reason=(
                        "synthetic-input test inference succeeded"
                        if self._n_features
                        else "inference succeeded with a guessed input width; "
                        "schema unknown, first real request will validate"
                    ),
                )
            except Exception as e:  # noqa: BLE001
                last_error = e
        return ValidationResult(
            status=ValidationStatus.PARTIAL,
            schema_confidence=SchemaConfidence.UNKNOWN,
            reason=f"could not infer a working input shape: {last_error}",
        )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
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
            import torch

            x = np.asarray(features, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if self._n_features is not None and x.shape[1] != self._n_features:
                raise AdapterInferenceError(
                    f"expected {self._n_features} features, got {x.shape[1]}"
                )
            with torch.no_grad():
                output = self._model(torch.from_numpy(x))

            preds, shape = _to_python(output)
            return PredictionResult(predictions=preds, raw_output_shape=shape)
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        version = None
        try:
            import torch

            version = torch.__version__
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="torchscript",
            model_type="TorchScript",
            task_type=self.task_type.value,
            n_features=self._n_features,
            framework_version=version,
        )


def _to_python(output):
    import torch

    if isinstance(output, dict):
        output = next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        output = output[0]

    if isinstance(output, torch.Tensor):
        array = output.detach().cpu().numpy()
        return array.tolist(), list(array.shape)
    return output, None
