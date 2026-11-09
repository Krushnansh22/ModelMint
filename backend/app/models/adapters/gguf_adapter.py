"""GGUF adapter (`.gguf` quantized LLM weights via llama-cpp-python).

This is the "stretch" format: it serves local text generation without a
Hugging Face bundle. GGUF files are native/untrusted, so the model runs in the
persistent sandboxed worker.

Payloads use the standard text contract (`text` or `prompt`) plus optional
`max_tokens`, `temperature`, and `top_p`. The response is a JSON-safe dict with
`generated_text` and token usage.
"""
from __future__ import annotations

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


class GGUFAdapter(ModelAdapter):
    task_type = TaskType.TEXT
    input_contract = InputContract.TEXT

    def __init__(self) -> None:
        self._llm = None
        self._llama_cpp = None
        self._n_ctx: Optional[int] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return file_path.lower().endswith(".gguf")

    def load(self, file_path: str) -> None:
        try:
            import llama_cpp  # lazy: optional dependency
        except ImportError as e:
            raise AdapterLoadError(
                "GGUF artifact detected but 'llama-cpp-python' is not installed. "
                "Install backend/requirements-formats.txt."
            ) from e

        try:
            self._n_ctx = 2048
            self._llm = llama_cpp.Llama(
                model_path=file_path,
                n_ctx=self._n_ctx,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            raise AdapterLoadError(f"failed to load GGUF model: {e}") from e
        self._llama_cpp = llama_cpp

    def unload(self) -> None:
        self._llm = None
        self._llama_cpp = None

    def validate(self) -> ValidationResult:
        if self._llm is None:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason="model not loaded",
            )
        try:
            self._llm("Hello", max_tokens=1, temperature=0.0)
            return ValidationResult(
                status=ValidationStatus.PASS,
                schema_confidence=SchemaConfidence.INFERRED,
                reason="synthetic-input test generation succeeded",
            )
        except Exception as e:  # noqa: BLE001
            return ValidationResult(
                status=ValidationStatus.PARTIAL,
                schema_confidence=SchemaConfidence.UNKNOWN,
                reason=f"synthetic generation failed: {e}",
            )

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return None

    def predict(self, payload: dict) -> PredictionResult:
        if self._llm is None:
            raise AdapterInferenceError("model not loaded")

        prompt = payload.get("prompt")
        if prompt is None:
            prompt = payload.get("text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AdapterInferenceError("payload must contain a non-empty 'prompt' or 'text' string")

        try:
            max_tokens = int(payload.get("max_tokens", 128))
            temperature = float(payload.get("temperature", 0.7))
            top_p = float(payload.get("top_p", 0.95))
            if max_tokens < 1 or max_tokens > 4096:
                raise AdapterInferenceError("'max_tokens' must be between 1 and 4096")

            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choices = output.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            return PredictionResult(
                predictions={
                    "generated_text": text,
                    "usage": output.get("usage", {}),
                }
            )
        except AdapterInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AdapterInferenceError(f"inference failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        version = None
        try:
            version = getattr(self._llama_cpp, "__version__", None)
        except Exception:  # noqa: BLE001
            pass
        return ModelMetadata(
            framework="gguf",
            model_type="GGUF",
            task_type=self.task_type.value,
            input_contract=self.input_contract.value,
            framework_version=version,
            extra={"n_ctx": self._n_ctx},
        )
