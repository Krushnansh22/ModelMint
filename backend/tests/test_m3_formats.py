"""M3 tests: declarative format registry, extension detection, task-aware
payload validation/examples, and batch prediction."""
from __future__ import annotations

import os
import uuid
import zipfile
from pathlib import Path

import pytest

from app.database import models as db_models
from app.models.adapters.base import AdapterDetectionError, AdapterLoadError
from app.models.adapters.registry import (
    adapter_for_framework,
    detect_adapter,
    framework_name,
    is_sandboxed,
    list_formats,
    supported_extensions,
)
from app.security.auth import create_api_key
from app.services.api_generator import build_example_request, validate_payload_against_schema
from app.models.adapters.base import AdapterInferenceError

FIXTURE = Path(__file__).parent / "test_adapters" / "fixtures" / "iris_rf.pkl"


# --- registry ---------------------------------------------------------------

def test_registry_covers_expected_frameworks():
    frameworks = {fmt["framework"] for fmt in list_formats()}
    assert {
        "sklearn",
        "onnx",
        "xgboost",
        "lightgbm",
        "catboost",
        "torchscript",
        "keras",
        "tflite",
        "hf",
        "gguf",
    } <= frameworks


def test_registry_marks_sandboxed_and_optional_dependencies():
    formats = {fmt["framework"]: fmt for fmt in list_formats()}

    assert formats["sklearn"]["sandboxed"] is True
    assert formats["torchscript"]["sandboxed"] is True
    assert formats["onnx"]["sandboxed"] is False

    # Optional packages must be declared, and `installed` must be a bool.
    assert formats["xgboost"]["optional_packages"] == ["xgboost"]
    assert isinstance(formats["xgboost"]["installed"], bool)
    assert is_sandboxed("sklearn") and is_sandboxed("torchscript")
    assert not is_sandboxed("onnx")


def test_supported_extensions_include_new_formats():
    extensions = supported_extensions()
    assert {
        ".pkl",
        ".joblib",
        ".onnx",
        ".json",
        ".ubj",
        ".txt",
        ".lgbm",
        ".model",
        ".cbm",
        ".pt",
        ".pth",
        ".torchscript",
        ".keras",
        ".h5",
        ".tflite",
        ".zip",
        ".gguf",
    } <= extensions


ONNX_FIXTURE = Path(__file__).parent / "test_adapters" / "fixtures" / "iris_rf.onnx"


@pytest.mark.parametrize(
    "path, framework",
    [
        (str(FIXTURE), "sklearn"),
        ("model.joblib", "sklearn"),
        (str(ONNX_FIXTURE), "onnx"),
        ("model.json", "xgboost"),
        ("model.ubj", "xgboost"),
        ("model.txt", "lightgbm"),
        ("model.model", "lightgbm"),
        ("model.cbm", "catboost"),
        ("model.pt", "torchscript"),
        ("model.pth", "torchscript"),
        ("model.torchscript", "torchscript"),
        ("model.keras", "keras"),
        ("model.h5", "keras"),
        ("model.tflite", "tflite"),
        ("model.gguf", "gguf"),
    ],
)
def test_detect_adapter_maps_extensions(path, framework):
    adapter_cls = detect_adapter(path)
    assert framework_name(adapter_cls) == framework
    assert adapter_for_framework(framework) is adapter_cls


def _hf_bundle_zip(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "bundle/config.json",
            '{"architectures": ["BertForSequenceClassification"], "model_type": "bert"}',
        )
        archive.writestr("bundle/model.safetensors", b"not-real-weights")
    return path


def test_detect_huggingface_zip_bundle(tmp_path):
    path = _hf_bundle_zip(tmp_path)
    adapter_cls = detect_adapter(str(path))
    assert framework_name(adapter_cls) == "hf"
    assert is_sandboxed("hf")


def test_huggingface_zip_without_config_is_not_detected(tmp_path):
    path = tmp_path / "plain.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "hello")
    with pytest.raises(AdapterDetectionError):
        detect_adapter(str(path))


@pytest.mark.parametrize(
    "adapter_name, path",
    [
        ("keras", "missing.keras"),
        ("tflite", "missing.tflite"),
        ("hf", "missing.zip"),
        ("gguf", "missing.gguf"),
    ],
)
def test_optional_adapters_fail_friendly_without_artifacts(adapter_name, path):
    """Loading an invalid artifact must raise the typed load error (not a
    bare ImportError/OSError) even when the optional package is absent."""
    adapter_cls = adapter_for_framework(adapter_name)
    with pytest.raises(AdapterLoadError):
        adapter_cls().load(path)


def test_detect_adapter_rejects_unknown_extension():
    with pytest.raises(AdapterDetectionError):
        detect_adapter("model.csv")


def test_adapter_for_unknown_framework_raises():
    with pytest.raises(AdapterDetectionError):
        adapter_for_framework("definitely-not-a-framework")


# --- task-aware validation / examples --------------------------------------

def _row(task_type: str, **overrides) -> db_models.Model:
    row = db_models.Model(
        id=str(uuid.uuid4()),
        name="t",
        slug=f"t-{uuid.uuid4().hex[:8]}",
        framework="onnx",
        file_path="/tmp/does-not-exist.onnx",
        status="READY",
        task_type=task_type,
        schema_confidence=overrides.pop("schema_confidence", "INFERRED"),
        input_schema=overrides.pop("input_schema", {"feature_count": 4, "fields": []}),
        version=1,
        **overrides,
    )
    return row


def test_text_validation_accepts_text_and_texts():
    row = _row("text")
    validate_payload_against_schema(row, {"text": "hello"})
    validate_payload_against_schema(row, {"texts": ["a", "b"]})


@pytest.mark.parametrize("payload", [{}, {"text": 3}, {"texts": [1, 2]}])
def test_text_validation_rejects_bad_payloads(payload):
    with pytest.raises(AdapterInferenceError):
        validate_payload_against_schema(_row("text"), payload)


def test_vision_validation_requires_image_base64():
    row = _row("vision")
    validate_payload_against_schema(row, {"image_base64": "aGVsbG8="})
    with pytest.raises(AdapterInferenceError):
        validate_payload_against_schema(row, {"features": [1, 2]})


def test_embedding_validation_accepts_text_or_features():
    row = _row("embedding")
    validate_payload_against_schema(row, {"text": "embed me"})
    validate_payload_against_schema(row, {"features": [1.0, 2.0]})
    with pytest.raises(AdapterInferenceError):
        validate_payload_against_schema(row, {})


def test_tabular_validation_still_enforces_feature_count():
    row = _row("tabular")
    validate_payload_against_schema(row, {"features": [1, 2, 3, 4]})
    with pytest.raises(AdapterInferenceError):
        validate_payload_against_schema(row, {"features": [1, 2]})


def test_task_aware_example_requests():
    assert "text" in build_example_request(_row("text"))
    assert "image_base64" in build_example_request(_row("vision"))
    tabular = build_example_request(_row("tabular"))
    assert len(tabular["features"]) == 4


# --- batch predict ----------------------------------------------------------

def _make_servable_model(db) -> db_models.Model:
    row = db_models.Model(
        id=str(uuid.uuid4()),
        name="iris",
        slug=f"iris-{uuid.uuid4().hex[:8]}",
        framework="sklearn",
        file_path=str(FIXTURE),
        file_size_bytes=os.path.getsize(FIXTURE),
        status="READY",
        schema_confidence="INFERRED",
        input_schema={"fields": [], "feature_count": 4},
        task_type="tabular",
        version=1,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_batch_predict_returns_one_result_per_instance(client, db):
    model = _make_servable_model(db)
    _, raw = create_api_key(db, name="batch", scopes=["predict"])

    res = client.post(
        f"/api/models/{model.id}/predict/batch",
        json={
            "instances": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [6.2, 3.4, 5.4, 2.3]},
            ]
        },
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["task_type"] == "tabular"


def test_batch_predict_rejects_empty_instances(client, db):
    model = _make_servable_model(db)
    _, raw = create_api_key(db, name="batch", scopes=["predict"])

    res = client.post(
        f"/api/models/{model.id}/predict/batch",
        json={"instances": []},
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 400


def test_batch_predict_reports_offending_instance(client, db):
    model = _make_servable_model(db)
    _, raw = create_api_key(db, name="batch", scopes=["predict"])

    res = client.post(
        f"/api/models/{model.id}/predict/batch",
        json={"instances": [{"features": [5.1, 3.5, 1.4, 0.2]}, {"features": [1, 2]}]},
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["instance_index"] == 1
