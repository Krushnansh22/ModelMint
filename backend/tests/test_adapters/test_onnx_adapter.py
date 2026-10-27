import os

from app.models.adapters.onnx import OnnxAdapter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "iris_rf.onnx")


def test_detect():
    assert OnnxAdapter.detect(FIXTURE) is True


def test_load_validate_predict():
    adapter = OnnxAdapter()
    adapter.load(FIXTURE)
    result = adapter.validate()
    assert result.status.value == "PASS"
    assert result.schema_confidence.value == "STRICT"

    schema = adapter.get_input_schema()
    assert schema is not None

    pred = adapter.predict({"features": [5.1, 3.5, 1.4, 0.2]})
    assert pred.predictions is not None

    adapter.unload()
