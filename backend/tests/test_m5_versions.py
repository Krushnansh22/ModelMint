"""M5 tests: model versions, promote/rollback, A/B traffic, and drift."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.database import models as db_models
from app.security.auth import create_api_key

FIXTURE = Path(__file__).parent / "test_adapters" / "fixtures" / "iris_rf.pkl"


def _key(db, *scopes):
    _, raw = create_api_key(db, name="m5", scopes=list(scopes))
    return raw


def _make_model(
    db,
    *,
    family_id: str | None = None,
    version: int = 1,
    is_active: bool = True,
    traffic_weight: float = 1.0,
) -> db_models.Model:
    model_id = str(uuid.uuid4())
    row = db_models.Model(
        id=model_id,
        name=f"iris-v{version}",
        slug=f"iris-{model_id[:8]}-v{version}",
        framework="sklearn",
        file_path=str(FIXTURE),
        file_size_bytes=FIXTURE.stat().st_size,
        status="READY",
        schema_confidence="INFERRED",
        input_schema={"fields": [], "feature_count": 4},
        task_type="tabular",
        version=version,
        family_id=family_id or model_id,
        is_active=is_active,
        traffic_weight=traffic_weight,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# --- upload / versioning ----------------------------------------------------

def test_upload_creates_active_v1_family(client, db):
    raw = _key(db, "models:write")
    with open(FIXTURE, "rb") as f:
        res = client.post(
            "/api/models",
            files={"file": ("iris.pkl", f)},
            headers={"X-API-Key": raw},
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["version"] == 1
    assert body["family_id"] == body["id"]
    assert body["is_active"] is True
    assert body["traffic_weight"] == 1.0


def test_upload_new_version_links_family(client, db):
    raw = _key(db, "models:write")
    with open(FIXTURE, "rb") as f:
        first = client.post(
            "/api/models",
            files={"file": ("iris.pkl", f)},
            headers={"X-API-Key": raw},
        )
    assert first.status_code == 201, first.text
    v1 = first.json()

    with open(FIXTURE, "rb") as f:
        second = client.post(
            f"/api/models/{v1['id']}/versions",
            files={"file": ("iris-v2.pkl", f)},
            headers={"X-API-Key": raw},
        )
    assert second.status_code == 201, second.text
    v2 = second.json()
    assert v2["version"] == 2
    assert v2["family_id"] == v1["family_id"]
    assert v2["is_active"] is False
    assert v2["traffic_weight"] == 0.0

    versions = client.get(f"/api/models/{v1['id']}/versions", headers={"X-API-Key": raw})
    assert versions.status_code == 200, versions.text
    body = versions.json()
    assert body["total"] == 2
    assert [v["version"] for v in body["versions"]] == [2, 1]


def test_promote_and_rollback(client, db):
    raw = _key(db, "models:write")
    v1 = _make_model(db, version=1, is_active=True, traffic_weight=1.0)
    v2 = _make_model(db, family_id=v1.family_id, version=2, is_active=False, traffic_weight=0.0)

    promoted = client.post(f"/api/models/{v2.id}/promote", headers={"X-API-Key": raw})
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["promoted"] == v2.id
    db.refresh(v1)
    db.refresh(v2)
    assert v2.is_active is True and v2.traffic_weight == 1.0
    assert v1.is_active is False and v1.traffic_weight == 0.0

    rolled = client.post(f"/api/models/{v2.id}/rollback", headers={"X-API-Key": raw})
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["rolled_back_to"] == v1.id
    db.refresh(v1)
    db.refresh(v2)
    assert v1.is_active is True and v1.traffic_weight == 1.0
    assert v2.is_active is False


def test_rollback_without_previous_version_is_rejected(client, db):
    raw = _key(db, "models:write")
    v1 = _make_model(db, version=1)
    res = client.post(f"/api/models/{v1.id}/rollback", headers={"X-API-Key": raw})
    assert res.status_code == 409


# --- A/B traffic ------------------------------------------------------------

def test_traffic_split_normalizes_and_serves_selected_version(client, db):
    write = _key(db, "models:write")
    predict = _key(db, "predict")
    v1 = _make_model(db, version=1, is_active=True, traffic_weight=1.0)
    v2 = _make_model(db, family_id=v1.family_id, version=2, is_active=False, traffic_weight=0.0)

    res = client.post(
        f"/api/models/{v1.id}/traffic",
        json={"weights": {v1.id: 1, v2.id: 3}},
        headers={"X-API-Key": write},
    )
    assert res.status_code == 200, res.text
    weights = {v["id"]: v["traffic_weight"] for v in res.json()["versions"]}
    assert weights[v1.id] == 0.25
    assert weights[v2.id] == 0.75

    # Force all traffic to v2, then assert the family alias serves v2.
    forced = client.post(
        f"/api/models/{v1.id}/traffic",
        json={"weights": {v2.id: 1}},
        headers={"X-API-Key": write},
    )
    assert forced.status_code == 200, forced.text

    pred = client.post(
        f"/api/families/{v1.family_id}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers={"X-API-Key": predict},
    )
    assert pred.status_code == 200, pred.text
    body = pred.json()
    assert body["model_id"] == v2.id
    assert body["version"] == 2
    assert body["family_id"] == v1.family_id

    # The inference log carries the version + an input summary for drift.
    logs = (
        db.query(db_models.InferenceLog)
        .filter(db_models.InferenceLog.model_id == v2.id)
        .all()
    )
    assert logs
    assert logs[-1].model_version == 2
    assert logs[-1].input_summary["kind"] == "tabular"


def test_traffic_rejects_unknown_version(client, db):
    write = _key(db, "models:write")
    v1 = _make_model(db, version=1)
    res = client.post(
        f"/api/models/{v1.id}/traffic",
        json={"weights": {"not-a-version": 1}},
        headers={"X-API-Key": write},
    )
    assert res.status_code == 400


# --- drift ------------------------------------------------------------------

def _log(db, model, mean, *, offset_minutes):
    db.add(
        db_models.InferenceLog(
            model_id=model.id,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=offset_minutes),
            status="SUCCESS",
            latency_ms=1.0,
            task_type="tabular",
            model_version=model.version,
            input_summary={
                "kind": "tabular",
                "n": 1,
                "mean": mean,
                "std": [1.0, 1.0, 1.0, 1.0],
                "min": mean,
                "max": mean,
            },
        )
    )


def test_drift_detects_shifted_feature(client, db):
    model = _make_model(db, version=1)
    # Older baseline: feature 0 around 0. Newer window: feature 0 around 10.
    for i in range(10):
        _log(db, model, [0.0, 0.0, 0.0, 0.0], offset_minutes=100 - i)
    for i in range(10):
        _log(db, model, [10.0, 0.0, 0.0, 0.0], offset_minutes=10 - i)
    db.commit()

    res = client.get(f"/api/models/{model.id}/drift?window=20")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "DRIFT"
    assert body["baseline_count"] == 10
    assert body["current_count"] == 10
    feature = body["features"][0]
    assert feature["feature"] == "feature_0"
    assert feature["drift"] is True
    assert feature["z_score"] > 3


def test_drift_reports_insufficient_data(client, db):
    model = _make_model(db, version=1)
    _log(db, model, [0.0, 0.0, 0.0, 0.0], offset_minutes=1)
    db.commit()

    res = client.get(f"/api/models/{model.id}/drift")
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "INSUFFICIENT_DATA"
