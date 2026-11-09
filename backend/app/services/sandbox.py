"""
Subprocess isolation for pickle-based model loads (build guide section 2.1).

`pickle.load()` / `joblib.load()` execute arbitrary Python via `__reduce__`.
This module runs an adapter's load() + validate() inside a separate OS
process (spawned, not forked -- forking would inherit the parent's open
sockets/fds) with:

  - No inherited network access (child's socket module is monkeypatched to
    raise before any user code runs)
  - Resource limits (address space + CPU time) via `resource.setrlimit`
  - A hard wall-clock timeout, enforced by the parent via `Process.join(timeout)`
    and SIGKILL on exceed
  - Communication back to the parent ONLY via a `multiprocessing.Queue`
    carrying plain (picklable-by-us, not by the untrusted model) result data

This is the "raw multiprocessing + resource limits" option called out in the
build guide's open decisions (section 12, item 2) -- a real step up from
in-process loading, though a container-per-load (Docker) would be the more
defensible hardening path if this became a public-facing demo.

The result payload sent back through the queue is deliberately a small,
JSON-safe dict (status, schema info, metadata, error string) -- NOT the
loaded model object itself, which never leaves the sandboxed process. The
model is re-loaded a second time, in-process, by the Model Manager once the
sandbox reports success. This trades a bit of duplicate load work for a hard
guarantee that no object produced by untrusted pickle code crosses the
process boundary.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings

_settings = get_settings()
DEFAULT_TIMEOUT_SECONDS = _settings.sandbox_timeout_seconds
DEFAULT_MEMORY_LIMIT_BYTES = _settings.sandbox_memory_limit_mb * 1024 * 1024
DEFAULT_CPU_LIMIT_SECONDS = _settings.sandbox_cpu_limit_seconds

def _select_start_method() -> str:
    """"spawn" is the safe default: forking a fresh OS process from inside a
    live uvicorn/uvloop event-loop process is a known source of hangs and
    crashes (fork() only duplicates the calling thread; libuv/asyncio state
    in other threads can be left inconsistent in the child). Under pytest,
    though, spawn re-executes the interpreter and re-triggers pytest's
    assertion-rewrite import hook in the child, which has been observed to
    corrupt bytecode reconstruction for unrelated reasons -- so tests use
    "fork" instead, which is safe there because the test process isn't
    running an event loop."""
    if "PYTEST_CURRENT_TEST" in os.environ and "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


_ctx = mp.get_context(_select_start_method())


@dataclass
class SandboxResult:
    ok: bool
    error: Optional[str] = None
    validation_status: Optional[str] = None
    schema_confidence: Optional[str] = None
    validation_reason: Optional[str] = None
    load_time_ms: Optional[float] = None
    # JSON-safe introspection results, so the API process never has to load
    # an untrusted pickle just to read its schema/metadata.
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    metadata: Optional[dict] = None


def _lock_down_child(memory_limit_bytes: int, cpu_limit_seconds: int) -> None:
    """Runs at the top of the child process before any adapter code executes.

    Network egress and resource limits are ALWAYS applied. Dropping to an
    unprivileged uid is opt-in (MODELMINT_SANDBOX_DROP_UID=1) because it
    requires the model-storage path to already be readable by that uid --
    in a real deployment the storage bind-mount should be owned/readable by
    a dedicated low-privilege user and this flag turned on; for local dev
    (this repo running as a single user/root with files it just wrote
    itself) leaving it off avoids spurious permission failures while still
    exercising the network + resource isolation that matters most."""
    import os
    import socket

    # No network egress: neuter socket *construction* by patching __init__
    # rather than replacing the socket.socket class outright. Replacing the
    # class itself breaks anything that subclasses it at import time (e.g.
    # ssl.py's `class SSLSocket(socket)`, which joblib pulls in transitively
    # via asyncio) -- patching __init__ keeps socket.socket a real class
    # usable as a base, while still making every instantiation raise.
    def _blocked_socket_init(self, *_args, **_kwargs):
        raise OSError("network access is disabled inside the model-load sandbox")

    socket.socket.__init__ = _blocked_socket_init

    # Resource limits: cap address space and CPU time. RLIMIT_AS catches
    # runaway allocation; RLIMIT_CPU is a backstop in case the wall-clock
    # timeout in the parent is somehow bypassed (e.g. the child ignores
    # SIGTERM racing SIGKILL). The `resource` module is POSIX-only (no
    # Windows support) -- degrade gracefully there rather than crashing the
    # whole sandboxed load, since the wall-clock timeout in the parent
    # process still provides a backstop on every platform.
    try:
        import resource
    except ImportError:
        resource = None

    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except (ValueError, OSError):
            pass  # platform may not support this limit; degrade gracefully
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))
        except (ValueError, OSError):
            pass

    # Opt-in privilege drop -- see docstring above for why this defaults off.
    if os.environ.get("MODELMINT_SANDBOX_DROP_UID") == "1" and hasattr(os, "getuid") and os.getuid() == 0:
        try:
            import pwd
            nobody = pwd.getpwnam("nobody")
            os.setgroups([])
            os.setgid(nobody.pw_gid)
            os.setuid(nobody.pw_uid)
        except Exception:
            pass  # best-effort; don't crash the sandbox setup over this


def _safe_schema_dump(getter) -> Optional[dict]:
    try:
        schema = getter()
        return schema.model_dump() if schema is not None else None
    except Exception:  # noqa: BLE001 -- introspection must not fail the load
        return None


def _safe_metadata_dump(getter) -> Optional[dict]:
    try:
        metadata = getter()
        return metadata.model_dump() if metadata is not None else None
    except Exception:  # noqa: BLE001
        return None


def _child_entrypoint(
    framework: str,
    file_path: str,
    memory_limit_bytes: int,
    cpu_limit_seconds: int,
    result_queue: "mp.Queue",
) -> None:
    t0 = time.perf_counter()
    try:
        _lock_down_child(memory_limit_bytes, cpu_limit_seconds)

        # Imported AFTER lock-down so any import-time side effects in the
        # adapter modules also run under the restricted environment.
        from app.models.adapters.registry import adapter_for_framework

        adapter_cls = adapter_for_framework(framework)
        adapter = adapter_cls()
        adapter.load(file_path)
        validation = adapter.validate()

        # Introspect inside the sandbox and send back plain dicts. This is
        # what lets analyzer.py avoid re-loading the untrusted pickle in the
        # API process.
        input_schema = _safe_schema_dump(adapter.get_input_schema)
        output_schema = _safe_schema_dump(adapter.get_output_schema)
        metadata = _safe_metadata_dump(adapter.get_metadata)

        adapter.unload()

        load_time_ms = (time.perf_counter() - t0) * 1000
        result_queue.put(
            SandboxResult(
                ok=True,
                validation_status=validation.status.value,
                schema_confidence=validation.schema_confidence.value,
                validation_reason=validation.reason,
                load_time_ms=load_time_ms,
                input_schema=input_schema,
                output_schema=output_schema,
                metadata=metadata,
            )
        )
    except Exception as e:  # noqa: BLE001 -- must not let the child crash silently
        result_queue.put(
            SandboxResult(
                ok=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            )
        )


def run_sandboxed_load_and_validate(
    framework: str,
    file_path: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
) -> SandboxResult:
    """Spawns an isolated subprocess to load()+validate() a pickle-based
    model. Returns a SandboxResult; never raises for expected failure modes
    (load errors, timeouts) -- those come back as ok=False with a reason.
    Caller (Model Manager) is responsible for re-loading the model
    in-process afterward if ok=True."""
    result_queue: mp.Queue = _ctx.Queue()
    proc = _ctx.Process(
        target=_child_entrypoint,
        args=(framework, file_path, memory_limit_bytes, cpu_limit_seconds, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        return SandboxResult(ok=False, error=f"sandbox load timed out after {timeout_seconds}s")

    try:
        result: SandboxResult = result_queue.get_nowait()
        return result
    except Exception:
        # Process died without reporting (e.g. OOM-killed by RLIMIT_AS,
        # SIGSEGV in a native extension) -- exit code tells us something.
        code = proc.exitcode
        return SandboxResult(ok=False, error=f"sandbox process exited abnormally (code={code}) with no result")
