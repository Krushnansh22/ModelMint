"""
Adapter detect() dispatch -- NOT the model registry (that's
app/models/registry.py, the Model Manager). This module answers one
question: "given a file on disk, which ModelAdapter class handles it?"

This is the ONLY place `if framework == ...`-style branching is allowed
(per the build guide's design principle in section 1).
"""
from app.models.adapters.base import ModelAdapter, AdapterDetectionError
from app.models.adapters.sklearn import SklearnAdapter
from app.models.adapters.onnx import OnnxAdapter

# Order matters only in the sense that each detect() must be cheap and
# mutually exclusive; ties are resolved by list order.
_ADAPTERS: list[type[ModelAdapter]] = [SklearnAdapter, OnnxAdapter]

# Which adapters require pickle-based loading (-> must go through the sandbox).
SANDBOXED_FRAMEWORKS = {"sklearn"}

_FRAMEWORK_NAMES = {
    SklearnAdapter: "sklearn",
    OnnxAdapter: "onnx",
}


def detect_adapter(file_path: str) -> type[ModelAdapter]:
    for adapter_cls in _ADAPTERS:
        if adapter_cls.detect(file_path):
            return adapter_cls
    raise AdapterDetectionError(f"no adapter recognized file: {file_path}")


def framework_name(adapter_cls: type[ModelAdapter]) -> str:
    return _FRAMEWORK_NAMES[adapter_cls]


def adapter_for_framework(framework: str) -> type[ModelAdapter]:
    for cls, name in _FRAMEWORK_NAMES.items():
        if name == framework:
            return cls
    raise AdapterDetectionError(f"unknown framework: {framework}")


def is_sandboxed(framework: str) -> bool:
    return framework in SANDBOXED_FRAMEWORKS
