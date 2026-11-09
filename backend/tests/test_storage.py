"""M4 tests: local storage backend, model path resolution, and /metrics."""
from __future__ import annotations

from pathlib import Path

from app.services.storage import LocalStorage, ensure_local_path


def test_local_storage_put_path_delete(tmp_path):
    storage = LocalStorage(str(tmp_path))
    src = tmp_path / "upload.bin"
    src.write_bytes(b"model-bytes")

    stored = storage.put_file(str(src), "abc123.pkl")
    assert Path(stored).read_bytes() == b"model-bytes"
    assert Path(stored).parent == tmp_path.resolve()
    assert not src.exists()  # moved, not copied

    assert storage.local_path("abc123.pkl") == stored

    storage.delete("abc123.pkl")
    assert not Path(stored).exists()
    # Deleting a missing key is a no-op.
    storage.delete("abc123.pkl")


def test_local_storage_ignores_path_traversal_keys(tmp_path):
    storage = LocalStorage(str(tmp_path))
    src = tmp_path / "upload.bin"
    src.write_bytes(b"x")

    stored = storage.put_file(str(src), "../../etc/passwd")
    assert Path(stored).parent == tmp_path.resolve()
    assert Path(stored).name == "passwd"


def test_ensure_local_path_passthrough_for_local_rows():
    class _Row:
        storage_backend = "local"
        storage_key = "abc.pkl"
        file_path = "/tmp/abc.pkl"

    assert ensure_local_path(_Row()) == "/tmp/abc.pkl"


def test_metrics_endpoint_exposes_prometheus_text(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "modelmint_models_total" in res.text
    assert "modelmint_inference_requests_total" in res.text
    assert res.headers["content-type"].startswith("text/plain")
