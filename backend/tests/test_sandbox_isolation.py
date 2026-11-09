"""Verifies the sandbox actually isolates: no network egress, timeout
enforcement, and that a legitimate model still loads successfully through it."""
import os
import pickle
import tempfile

from app.services.sandbox import run_sandboxed_load_and_validate

FIXTURE = os.path.join(os.path.dirname(__file__), "test_adapters", "fixtures", "iris_rf.pkl")


def test_legitimate_model_loads_through_sandbox():
    result = run_sandboxed_load_and_validate("sklearn", FIXTURE)
    assert result.ok is True
    assert result.schema_confidence in ("STRICT", "INFERRED", "UNKNOWN")


def test_network_access_is_blocked_inside_sandbox():
    """A pickle whose __reduce__ tries to open a socket should fail cleanly
    inside the sandbox rather than actually reaching the network."""

    class _NetworkAttempt:
        def __reduce__(self):
            code = (
                "import socket; "
                "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
                "s.connect(('example.com', 80))"
            )
            return (exec, (code,))

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(_NetworkAttempt(), f)
        malicious_path = f.name

    try:
        result = run_sandboxed_load_and_validate("sklearn", malicious_path)
        # The load must fail (network blocked -> exception during unpickle),
        # and critically must NOT report ok=True.
        assert result.ok is False
    finally:
        os.unlink(malicious_path)


def test_sandboxed_adapter_proxy_predicts_and_unloads():
    """Serving path (M2): the model lives in a worker process; the parent
    only sends payloads and receives JSON-safe results."""
    from app.services.sandboxed_adapter import SandboxedAdapterProxy

    proxy = SandboxedAdapterProxy("sklearn")
    proxy.load(FIXTURE)
    try:
        assert proxy.validate().status.value in ("PASS", "PARTIAL")
        schema = proxy.get_input_schema()
        assert schema is not None and schema.feature_count == 4

        result = proxy.predict({"features": [5.1, 3.5, 1.4, 0.2]})
        assert len(result.predictions) == 1

        worker = proxy._proc
        assert worker is not None and worker.is_alive()
    finally:
        proxy.unload()

    assert proxy._proc is None
    assert not worker.is_alive()


def test_timeout_is_enforced():
    class _InfiniteLoop:
        def __reduce__(self):
            return (exec, ("while True: pass",))

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(_InfiniteLoop(), f)
        hanging_path = f.name

    try:
        result = run_sandboxed_load_and_validate("sklearn", hanging_path, timeout_seconds=3)
        assert result.ok is False
        assert "timed out" in (result.error or "")
    finally:
        os.unlink(hanging_path)
