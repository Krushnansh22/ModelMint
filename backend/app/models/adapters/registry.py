"""
Adapter detect() dispatch -- NOT the model registry (that's
app/models/registry.py, the Model Manager). This module answers one
question: "given a file on disk, which ModelAdapter class handles it?"

This is the ONLY place `if framework == ...`-style branching is allowed
(per the build guide's design principle in section 1). M3 replaces the old
hand-maintained tables with declarative `AdapterSpec` entries; adding a
format means adding one spec here (plus its adapter module).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

from app.models.adapters.base import AdapterDetectionError, ModelAdapter
from app.models.adapters.catboost_adapter import CatBoostAdapter
from app.models.adapters.gguf_adapter import GGUFAdapter
from app.models.adapters.hf_adapter import HuggingFaceBundleAdapter
from app.models.adapters.keras_adapter import KerasAdapter
from app.models.adapters.lightgbm_adapter import LightGBMAdapter
from app.models.adapters.onnx import OnnxAdapter
from app.models.adapters.sklearn import SklearnAdapter
from app.models.adapters.tflite_adapter import TFLiteAdapter
from app.models.adapters.torchscript_adapter import TorchScriptAdapter
from app.models.adapters.xgboost_adapter import XGBoostAdapter


@dataclass(frozen=True)
class AdapterSpec:
    cls: type[ModelAdapter]
    name: str
    extensions: tuple[str, ...]
    task_types: tuple[str, ...]
    sandboxed: bool
    optional_packages: tuple[str, ...] = ()
    description: str = ""

    @property
    def installed(self) -> bool:
        return all(_package_installed(pkg) for pkg in self.optional_packages)


def _package_installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# Order matters only in the sense that each detect() must be cheap and
# mutually exclusive; ties are resolved by list order.
_SPECS: list[AdapterSpec] = [
    AdapterSpec(
        cls=SklearnAdapter,
        name="sklearn",
        extensions=(".pkl", ".joblib"),
        task_types=("tabular",),
        sandboxed=True,
        description="scikit-learn estimator saved with joblib/pickle",
    ),
    AdapterSpec(
        cls=OnnxAdapter,
        name="onnx",
        extensions=(".onnx",),
        task_types=("tabular", "vision"),
        sandboxed=False,
        description="ONNX Runtime graph",
    ),
    AdapterSpec(
        cls=XGBoostAdapter,
        name="xgboost",
        extensions=(".json", ".ubj", ".xgb"),
        task_types=("tabular",),
        sandboxed=False,
        optional_packages=("xgboost",),
        description="XGBoost Booster (JSON/UBJSON/legacy binary)",
    ),
    AdapterSpec(
        cls=LightGBMAdapter,
        name="lightgbm",
        extensions=(".txt", ".lgbm", ".model"),
        task_types=("tabular",),
        sandboxed=False,
        optional_packages=("lightgbm",),
        description="LightGBM Booster text model",
    ),
    AdapterSpec(
        cls=CatBoostAdapter,
        name="catboost",
        extensions=(".cbm",),
        task_types=("tabular",),
        sandboxed=False,
        optional_packages=("catboost",),
        description="CatBoost model",
    ),
    AdapterSpec(
        cls=TorchScriptAdapter,
        name="torchscript",
        extensions=(".pt", ".pth", ".torchscript"),
        task_types=("tabular", "vision"),
        sandboxed=True,
        optional_packages=("torch",),
        description="TorchScript (traced/scripted PyTorch); raw state_dicts need a manifest",
    ),
    AdapterSpec(
        cls=KerasAdapter,
        name="keras",
        extensions=(".keras", ".h5"),
        task_types=("tabular", "vision"),
        sandboxed=True,
        optional_packages=("tensorflow",),
        description="Keras 3 / tf.keras SavedModel (no custom_objects)",
    ),
    AdapterSpec(
        cls=TFLiteAdapter,
        name="tflite",
        extensions=(".tflite",),
        task_types=("tabular", "vision"),
        sandboxed=True,
        optional_packages=("tensorflow",),
        description="TensorFlow Lite flatbuffer (tflite-runtime or tensorflow)",
    ),
    AdapterSpec(
        cls=HuggingFaceBundleAdapter,
        name="hf",
        extensions=(".zip",),
        task_types=("text", "vision", "embedding"),
        sandboxed=True,
        optional_packages=("transformers",),
        description="Hugging Face model directory zipped with config.json + weights (no trust_remote_code)",
    ),
    AdapterSpec(
        cls=GGUFAdapter,
        name="gguf",
        extensions=(".gguf",),
        task_types=("text",),
        sandboxed=True,
        optional_packages=("llama_cpp",),
        description="GGUF quantized LLM via llama-cpp-python (stretch format)",
    ),
]

_ADAPTERS: list[type[ModelAdapter]] = [spec.cls for spec in _SPECS]
_FRAMEWORK_NAMES: dict[type[ModelAdapter], str] = {spec.cls: spec.name for spec in _SPECS}
_SPEC_BY_NAME: dict[str, AdapterSpec] = {spec.name: spec for spec in _SPECS}

# Which adapters require process isolation (untrusted code / pickle / native).
SANDBOXED_FRAMEWORKS = {spec.name for spec in _SPECS if spec.sandboxed}


def detect_adapter(file_path: str) -> type[ModelAdapter]:
    for adapter_cls in _ADAPTERS:
        if adapter_cls.detect(file_path):
            return adapter_cls
    raise AdapterDetectionError(f"no adapter recognized file: {file_path}")


def framework_name(adapter_cls: type[ModelAdapter]) -> str:
    return _FRAMEWORK_NAMES[adapter_cls]


def adapter_for_framework(framework: str) -> type[ModelAdapter]:
    spec = _SPEC_BY_NAME.get(framework)
    if spec is None:
        raise AdapterDetectionError(f"unknown framework: {framework}")
    return spec.cls


def spec_for_framework(framework: str) -> Optional[AdapterSpec]:
    return _SPEC_BY_NAME.get(framework)


def is_sandboxed(framework: str) -> bool:
    return framework in SANDBOXED_FRAMEWORKS


def supported_extensions() -> set[str]:
    return {ext for spec in _SPECS for ext in spec.extensions}


def list_formats() -> list[dict]:
    """Format catalog for the UI / `GET /api/formats`."""
    return [
        {
            "framework": spec.name,
            "extensions": list(spec.extensions),
            "task_types": list(spec.task_types),
            "sandboxed": spec.sandboxed,
            "optional_packages": list(spec.optional_packages),
            "installed": spec.installed,
            "description": spec.description,
        }
        for spec in _SPECS
    ]
