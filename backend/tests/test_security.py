"""M1 security tests: key hashing, scopes, revocation, model binding,
rate limiting, and upload auth."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from app.database import models as db_models
from app.security.auth import create_api_key, revoke_api_key

FIXTURE = Path(__file__).parent / "test_adapters" / "fixtures" / "iris_rf.pkl"


def _make_model(db, *, framework="sklearn", status="READY", slug=None):
    row = db_models.Model(
        id=str(uuid.uuid4()),
        name="iris",
        slug=slug or f"iris-{uuid.uuid4().hex[:8]}",
        framework=framework,
        file_path=str(FIXTURE),
        file_size_bytes=os.path.getsize(FIXTURE) if FIXTURE.exists() else 1,
        status=status,
        schema_confidence="INFERRED",
        input_schema={"fields": [], "feature_count": 4},
        output_schema=None,
        task_type="tabular",
        version=1,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _key(db, scopes, model_id=None):
    row, raw = create_api_key(db, name="test-key", scopes=scopes, model_id=model_id)
    return row, raw


def _auth(raw):
    return {"X-API-Key": raw}


def test_key_is_stored_hashed_not_plaintext(db):
    row, raw = _key(db, ["predict"])
    assert row.key_hash != raw
    assert raw not in row.key_hash
    assert row.prefix == raw[:16]
    assert len(row.key_hash) == 64


def test_predict_requires_api_key(client, db):
    model = _make_model(db)
    res = client.post(f"/api/models/{model.id}/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert res.status_code == 401
    assert "WWW-Authenticate" in res.headers


def test_predict_rejects_wrong_scope(client, db):
    model = _make_model(db)
    _, raw = _key(db, ["models:read"])
    res = client.post(
        f"/api/models/{model.id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers=_auth(raw),
    )
    assert res.status_code == 403


def test_predict_succeeds_with_predict_scope(client, db):
    model = _make_model(db)
    _, raw = _key(db, ["predict"])
    res = client.post(
        f"/api/models/{model.id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers=_auth(raw),
    )
    assert res.status_code == 200, res.text
    assert "predictions" in res.json()
    assert res.json()["task_type"] == "tabular"


def test_predict_increments_key_request_count(client, db):
    model = _make_model(db)
    key_row, raw = _key(db, ["predict"])
    client.post(
        f"/api/models/{model.id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers=_auth(raw),
    )
    db.refresh(key_row)
    assert key_row.request_count == 1


def test_revoked_key_is_rejected(client, db):
    model = _make_model(db)
    key_row, raw = _key(db, ["predict"])
    revoke_api_key(db, key_row)
    res = client.post(
        f"/api/models/{model.id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers=_auth(raw),
    )
    assert res.status_code == 401


def test_model_bound_key_cannot_call_other_model(client, db):
    model_a = _make_model(db)
    model_b = _make_model(db)
    _, raw = _key(db, ["predict"], model_id=model_a.id)
    res = client.post(
        f"/api/models/{model_b.id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers=_auth(raw),
    )
    assert res.status_code == 403


def test_upload_requires_models_write(client, db):
    _, read_key = _key(db, ["models:read"])
    res = client.post("/api/models", files={"file": ("x.pkl", b"not a model")}, headers=_auth(read_key))
    assert res.status_code == 403

    res = client.post("/api/models", files={"file": ("x.pkl", b"not a model")})
    assert res.status_code == 401


def test_upload_rejects_unsupported_extension(client, db):
    _, raw = _key(db, ["models:write"])
    res = client.post(
        "/api/models",
        files={"file": ("model.csv", b"hello")},
        headers=_auth(raw),
    )
    assert res.status_code == 400


def test_formats_endpoint(client):
    res = client.get("/api/formats")
    assert res.status_code == 200
    frameworks = {f["framework"] for f in res.json()["formats"]}
    assert {"sklearn", "onnx"} <= frameworks


def test_rate_limit_returns_429(client, db, monkeypatch):
    from app.security import rate_limit as rl

    class _Settings:
        rate_limit_predict_per_min = 1
        redis_url = None

    monkeypatch.setattr(rl, "get_settings", lambda: _Settings())
    rl.reset_limiter()

    model = _make_model(db)
    _, raw = _key(db, ["predict"])
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}

    first = client.post(f"/api/models/{model.id}/predict", json=payload, headers=_auth(raw))
    second = client.post(f"/api/models/{model.id}/predict", json=payload, headers=_auth(raw))

    assert first.status_code == 200, first.text
    assert second.status_code == 429
    assert "Retry-After" in second.headers
