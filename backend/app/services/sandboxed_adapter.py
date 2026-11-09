"""
Persistent sandboxed adapter worker (M2).

The upload-time sandbox in `sandbox.py` protects analysis. This module
protects *serving*: a pickle/torch/Keras/HF model is loaded once in a
long-lived child process with the same no-network + RLIMIT restrictions, and
the API process only exchanges small JSON-safe messages with it. The model
object never enters the API process.

One worker per resident model. `ModelManager` creates these on demand and
`unload()` kills the worker on LRU eviction.
"""
from __future__ import annotations

import threading
from typing import Optional

from app.config import get_settings
from app.models.adapters.base import AdapterInferenceError, AdapterLoadError, ModelAdapter
from app.models.schemas import (
    ModelMetadata,
    PredictionResult,
    SchemaConfidence,
    SchemaDescriptor,
    ValidationResult,
    ValidationStatus,
)
from app.services.sandbox import _ctx, _lock_down_child


def _dump_schema(getter):
    try:
        schema = getter()
        return schema.model_dump() if schema is not None else None
    except Exception:  # noqa: BLE001
        return None


def _dump_metadata(getter):
    try:
        metadata = getter()
        return metadata.model_dump() if metadata is not None else None
    except Exception:  # noqa: BLE001
        return None


def _worker_main(
    framework: str,
    file_path: str,
    conn,
    memory_limit_bytes: int,
    cpu_limit_seconds: int,
) -> None:
    adapter = None
    try:
        _lock_down_child(memory_limit_bytes, cpu_limit_seconds)

        # Imported after lock-down so import-time side effects run restricted.
        from app.models.adapters.registry import adapter_for_framework

        adapter = adapter_for_framework(framework)()
        adapter.load(file_path)
        validation = adapter.validate()

        conn.send(
            {
                "type": "ready",
                "ok": True,
                "validation_status": validation.status.value,
                "schema_confidence": validation.schema_confidence.value,
                "validation_reason": validation.reason,
                "input_schema": _dump_schema(adapter.get_input_schema),
                "output_schema": _dump_schema(adapter.get_output_schema),
                "metadata": _dump_metadata(adapter.get_metadata),
            }
        )

        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            if message.get("type") == "shutdown":
                break
            if message.get("type") != "predict":
                continue
            try:
                result = adapter.predict(message.get("payload") or {})
                conn.send(
                    {
                        "type": "result",
                        "ok": True,
                        "predictions": result.predictions,
                        "raw_output_shape": result.raw_output_shape,
                    }
                )
            except Exception as e:  # noqa: BLE001 -- report, don't kill the worker
                conn.send(
                    {
                        "type": "result",
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
    except Exception as e:  # noqa: BLE001 -- startup failure
        try:
            conn.send({"type": "ready", "ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if adapter is not None:
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class SandboxedAdapterProxy(ModelAdapter):
    """ModelAdapter facade backed by a persistent sandboxed worker process."""

    def __init__(self, framework: str) -> None:
        self._framework = framework
        self._conn = None
        self._proc = None
        self._lock = threading.Lock()
        self._validation = ValidationResult(
            status=ValidationStatus.FAIL,
            schema_confidence=SchemaConfidence.UNKNOWN,
            reason="model not loaded",
        )
        self._input_schema: Optional[SchemaDescriptor] = None
        self._output_schema: Optional[SchemaDescriptor] = None
        self._metadata: Optional[ModelMetadata] = None

    @staticmethod
    def detect(file_path: str) -> bool:
        return False

    def load(self, file_path: str) -> None:
        settings = get_settings()
        parent_conn, child_conn = _ctx.Pipe(duplex=True)
        proc = _ctx.Process(
            target=_worker_main,
            args=(
                self._framework,
                file_path,
                child_conn,
                settings.sandbox_memory_limit_mb * 1024 * 1024,
                settings.sandbox_cpu_limit_seconds,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        if not parent_conn.poll(settings.sandbox_timeout_seconds):
            self._kill(proc, parent_conn)
            raise AdapterLoadError("sandboxed worker failed to start (timeout)")

        try:
            message = parent_conn.recv()
        except EOFError as e:
            self._kill(proc, parent_conn)
            raise AdapterLoadError("sandboxed worker died during startup") from e

        if not message.get("ok"):
            self._kill(proc, parent_conn)
            raise AdapterLoadError(message.get("error") or "sandboxed worker failed to load model")

        self._conn = parent_conn
        self._proc = proc
        self._validation = ValidationResult(
            status=ValidationStatus(message["validation_status"]),
            schema_confidence=SchemaConfidence(message["schema_confidence"]),
            reason=message.get("validation_reason") or "",
        )
        if message.get("input_schema"):
            self._input_schema = SchemaDescriptor(**message["input_schema"])
        if message.get("output_schema"):
            self._output_schema = SchemaDescriptor(**message["output_schema"])
        if message.get("metadata"):
            self._metadata = ModelMetadata(**message["metadata"])

    def unload(self) -> None:
        with self._lock:
            conn, proc = self._conn, self._proc
            self._conn = None
            self._proc = None
        if conn is not None:
            try:
                conn.send({"type": "shutdown"})
            except Exception:  # noqa: BLE001
                pass
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._join_kill(proc)

    def validate(self) -> ValidationResult:
        return self._validation

    def get_input_schema(self) -> Optional[SchemaDescriptor]:
        return self._input_schema

    def get_output_schema(self) -> Optional[SchemaDescriptor]:
        return self._output_schema

    def predict(self, payload: dict) -> PredictionResult:
        timeout = get_settings().inference_timeout_seconds
        with self._lock:
            if self._conn is None:
                raise AdapterInferenceError("model is not loaded")

            try:
                self._conn.send({"type": "predict", "payload": payload})
            except Exception as e:  # noqa: BLE001
                self._teardown_locked()
                raise AdapterInferenceError(f"failed to reach sandboxed worker: {e}") from e

            if not self._conn.poll(timeout):
                self._teardown_locked()
                raise AdapterInferenceError(f"inference timed out after {timeout}s")

            try:
                message = self._conn.recv()
            except EOFError as e:
                self._teardown_locked()
                raise AdapterInferenceError("sandboxed worker died during inference") from e

            if not message.get("ok"):
                raise AdapterInferenceError(message.get("error") or "inference failed")

            return PredictionResult(
                predictions=message.get("predictions"),
                raw_output_shape=message.get("raw_output_shape"),
            )

    def get_metadata(self) -> ModelMetadata:
        return self._metadata or ModelMetadata(framework=self._framework)

    # --- lifecycle helpers ------------------------------------------------
    def _teardown_locked(self) -> None:
        conn, proc = self._conn, self._proc
        self._conn = None
        self._proc = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._join_kill(proc)

    @staticmethod
    def _join_kill(proc) -> None:
        if proc is None:
            return
        try:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _kill(proc, conn) -> None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        SandboxedAdapterProxy._join_kill(proc)
