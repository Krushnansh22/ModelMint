"""Hugging Face bundle adapter (a directory or `.zip` with `config.json`).

A bundle is a saved `transformers` model directory:

    config.json
    tokenizer.json / tokenizer_config.json / vocab.txt ...
    model.safetensors | pytorch_model.bin | tf_model.h5 | model.onnx

Uploads are single files, so the supported upload shape is a `.zip` of that
directory. Local directories are also accepted (useful for tests and for
mounting a bundle directly). We never pass `trust_remote_code=True`; a bundle
that requires custom Python is rejected rather than executed.

Served through the persistent sandboxed worker because transformers model
loading is native/untrusted.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import Optional

from app.models.adapters.base import AdapterInferenceError, AdapterLoadError, ModelAdapter
from app.models.schemas import (
    InputContract,
    ModelMetadata,
    PredictionResult,
    SchemaConfidence,
    SchemaDescriptor,
    TaskType,
    ValidationResult,
    ValidationStatus,
)

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".h5", ".onnx", ".msgpack")

# pipeline task -> (task type, input contract)
_PIPELINE_TASKS = {
    "text-classification": (TaskType.TEXT, InputContract.TEXT),
    "token-classification": (TaskType.TEXT, InputContract.TEXT),
    "question-answering": (TaskType.TEXT, InputContract.TEXT),
    "text-generation": (TaskType.TEXT, InputContract.TEXT),
    "text2text-generation": (TaskType.TEXT, InputContract.TEXT),
    "summarization": (TaskType.TEXT, InputContract.TEXT),
    "translation": (TaskType.TEXT, InputContract.TEXT),
    "fill-mask": (TaskType.TEXT, InputContract.TEXT),
    "zero-shot-classification": (TaskType.TEXT, InputContract.TEXT),
    "feature-extraction": (TaskType.EMBEDDING, InputContract.TEXT),
    "sentence-similarity": (TaskType.EMBEDDING, InputContract.TEXT),
    "image-classification": (TaskType.VISION, InputContract.IMAGE),
    "image-segmentation": (TaskType.VISION, InputContract.IMAGE),
    "object-detection": (TaskType.VISION, InputContract.IMAGE),
    "zero-shot-image-classification": (TaskType.VISION, InputContract.IMAGE),
}

_ARCHITECTURE_TASKS = {
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForQuestionAnswering": "question-answering",
    "ForCausalLM": "text-generation",
    "ForSeq2SeqLM": "text2text-generation",
    "ForConditionalGeneration": "text2text-generation",
    "ForMaskedLM": "fill-mask",
    "ForImageClassification": "image-classification",
    "ForImageSegmentation": "image-segmentation",
    "ForObjectDetection": "object-detection",
}


class HuggingFaceBundleAdapter(ModelAdapter):
    task_type = TaskType.TEXT
    input_contract = InputContract.TEXT

    def __init__(self) -> None:
        self._pipe = None
        self._pipeline_task: Optional[str] = None
        self._model_dir: Optional[str] = None
        self._tmpdir: Optional[str] = None
        self._config: dict = {}
        self._transformers_version: Optional[str] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        if os.path.isdir(file_path):
            return os.path.isfile(os.path.join(file_path, "config.json"))
        if not file_path.lower().endswith(".zip"):
            return False
        try:
            with zipfile.ZipFile(file_path) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile):
            return False
        configs = [n for n in names if n.endswith("config.json") and n.count("/") <= 1]
        weights = [n for n in names if n.endswith(_WEIGHT_SUFFIXES)]
        return bool(configs and weights)

    def load(self, file_path: str) -> None:
        try:
            import transformers  # lazy: optional dependency
        except ImportError as e:
            raise AdapterLoadError(
                "Hugging Face bundle detected but the 'transformers' package is "
                "not installed. Install backend/requirements-formats.txt."
            ) from e

        self._transformers_version = getattr(transformers, "__version__", None)
        try:
            self._model_dir = self._resolve_model_dir(file_path)
            with open(os.path.join(self._model_dir, "config.json")) as f:
                self._config = json.load(f)
            self._pipeline_task = _infer_pipeline_task(self._config)
            self.task_type, self.input_contract = _PIPELINE_TASKS.get(
                self._pipeline_task, (TaskType.TEXT, InputContract.TEXT)
            )
            self._pipe = _build_pipeline(
                transformers, self._pipeline_task, self._model_dir
            )
        except AdapterLoadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load Hugging Face bundle: {e}") from e

    def _resolve_model_dir(self, file_path: str) -> str:
        if os.path.isdir(file_path):
            return file_path
        self._tmpdir = tempfile.mkdtemp(prefix="modelmint_hf_")
        _safe_extract(file_path, self._tmpdir)
        for root, _dirs, files in os.walk(self._tmpdir):
            if "config.json" in files:
                return root
        raise AdapterLoadError("Hugging Face zip does not contain a config.json")

    def unload(self) -> None:
        self._pipe = None
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        self._model_dir = None
        self._config = {}

    def validate(self) -> ValidationResult:
        if self._pipe is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        try:
            if self.task_type == TaskType.VISION:
                self._pipe(_blank_image())
            elif self._pipeline_task == "text-generation":
                self._pipe("Hello", max_new_tokens=1)
            else:
                self._pipe("ModelMint validation.")
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=SchemaConfidence.INFERRED,
                reason="synthetic-input test inference succeeded",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason=f"synthetic inference failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._pipe is None:
            raise AdapterInferenceError("model not loaded")
        try:
            if self.task_type == TaskType.VISION:
                image_b64 = payload.get("image_base64")
                if not isinstance(image_b64, str) or not image_b64.strip():
                    raise AdapterInferenceError(
                        "payload must contain a non-empty 'image_base64' string"
                    )
                output = self._pipe(_decode_image(image_b64))
            else:
                text = payload.get("text")
                texts = payload.get("texts")
                if text is None and texts is None:
                    raise AdapterInferenceError("payload must contain 'text' or 'texts'")
                value = texts if texts is not None else text
                kwargs = {}
                if self._pipeline_task == "text-generation":
                    kwargs["max_new_tokens"] = int(payload.get("max_tokens", 128))
                    if "temperature" in payload:
                        kwargs["temperature"] = float(payload["temperature"])
                output = self._pipe(value, **kwargs)
            return PredictionResult(predictions=_jsonable(output))
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        architectures = self._config.get("architectures") or []
        id2label = self._config.get("id2label") or {}
        n_classes = len(id2label) if isinstance(id2label, dict) and id2label else None
        return ModelMetadata(
            framework="hf",
            model_type=architectures[0] if architectures else self._config.get("model_type"),
            task_type=self.task_type.value,
            input_contract=self.input_contract.value,
            n_classes=n_classes,
            framework_version=self._transformers_version,
            extra={
                "pipeline_task": self._pipeline_task,
                "architectures": architectures,
                "model_type": self._config.get("model_type"),
            },
        )


def _infer_pipeline_task(config: dict) -> str:
    tag = config.get("pipeline_tag")
    if isinstance(tag, str) and tag in _PIPELINE_TASKS:
        return tag
    architectures = config.get("architectures") or []
    if isinstance(architectures, list):
        for architecture in architectures:
            for suffix, task in _ARCHITECTURE_TASKS.items():
                if str(architecture).endswith(suffix):
                    return task
    return "feature-extraction"


def _build_pipeline(transformers, task: str, model_dir: str):
    kwargs = {"model": model_dir, "device": -1, "local_files_only": True}
    if task not in ("image-classification", "image-segmentation", "object-detection"):
        kwargs["tokenizer"] = model_dir
    return transformers.pipeline(task, **kwargs)


def _safe_extract(zip_path: str, target_dir: str) -> None:
    """Extract a zip while rejecting path traversal members."""
    target_root = os.path.realpath(target_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = os.path.realpath(os.path.join(target_dir, member.filename))
            if destination != target_root and not destination.startswith(target_root + os.sep):
                raise AdapterLoadError("Hugging Face zip contains an unsafe path")
            archive.extract(member, target_dir)


def _decode_image(image_b64: str):
    try:
        from PIL import Image  # lazy: optional
    except ImportError as e:  # pragma: no cover
        raise AdapterInferenceError(
            "image payloads require Pillow; install backend/requirements-formats.txt"
        ) from e
    try:
        raw = base64.b64decode(image_b64, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise AdapterInferenceError(f"could not decode 'image_base64': {e}") from e


def _blank_image():
    try:
        from PIL import Image  # lazy: optional
    except ImportError as e:  # pragma: no cover
        raise AdapterInferenceError("image payloads require Pillow") from e
    return Image.new("RGB", (224, 224), color=(0, 0, 0))


def _jsonable(value):
    """Recursively convert numpy scalars/arrays to plain JSON-safe values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
