"""Upload hardening tests: checksums, duplicates, empty files, size caps,
and configurable read auth."""
from __future__ import annotations

from pathlib import Path

from app.security.auth import create_api_key

FIXTURE = Path(__file__).parent / "test_adapters" / "fixtures" / "iris_rf.pkl"


def _admin(db):
    _, raw = create_api_key(db, name="admin", scopes=["admin"])
    return raw


def _write_key(db):
    _, raw = create_api_key(db, name="writer", scopes=["models:write"])
    return raw


def test_upload_stores_checksum_and_flags_duplicate(client, db):
    raw = _write_key(db)
    headers = {"X-API-Key": raw}

    with open(FIXTURE, "rb") as f:
        first = client.post("/api/models", files={"file": ("iris.pkl", f)}, headers=headers)
    assert first.status_code == 201, first.text
    assert len(first.json()["checksum"]) == 64

    with open(FIXTURE, "rb") as f:
        second = client.post("/api/models", files={"file": ("iris-copy.pkl", f)}, headers=headers)
    assert second.status_code == 201, second.text
    assert second.json()["duplicate_of"]["id"] == first.json()["id"]


def test_upload_rejects_empty_file(client, db):
    raw = _write_key(db)
    res = client.post(
        "/api/models",
        files={"file": ("empty.pkl", b"")},
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 400
    assert "empty" in res.text.lower()


def test_upload_size_cap(client, db, monkeypatch):
    from app.api import models as models_api

    class _Small:
        max_upload_bytes = 16
        max_upload_mb = 0
        storage_backend = "local"

    monkeypatch.setattr(models_api, "settings", _Small())

    raw = _write_key(db)
    res = client.post(
        "/api/models",
        files={"file": ("big.pkl", b"x" * 64)},
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 413


def test_audit_log_records_key_creation(client, db):
    admin = _admin(db)
    res = client.post(
        "/api/keys",
        json={"name": "child", "scopes": ["predict"]},
        headers={"X-API-Key": admin},
    )
    assert res.status_code == 201

    audit = client.get("/api/keys/audit", headers={"X-API-Key": admin})
    assert audit.status_code == 200
    assert any(row["action"] == "key.create" for row in audit.json())


def test_reads_can_require_auth(client, db, monkeypatch):
    from app.security import auth as auth_mod

    class _Settings:
        require_auth_for_reads = True
        api_key_pepper = "test-pepper"

    monkeypatch.setattr(auth_mod, "get_settings", lambda: _Settings())

    assert client.get("/api/models").status_code == 401

    _, raw = create_api_key(db, name="reader", scopes=["models:read"])
    assert client.get("/api/models", headers={"X-API-Key": raw}).status_code == 200
