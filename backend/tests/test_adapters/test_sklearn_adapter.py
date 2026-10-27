import os

from app.models.adapters.sklearn import SklearnAdapter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "iris_rf.pkl")


def test_detect():
    assert SklearnAdapter.detect(FIXTURE) is True


def test_load_validate_predict():
    adapter = SklearnAdapter()
    adapter.load(FIXTURE)
    result = adapter.validate()
    assert result.status.value in ("PASS", "PARTIAL")

    schema = adapter.get_input_schema()
    assert schema is not None
    assert schema.feature_count == 4

    pred = adapter.predict({"features": [5.1, 3.5, 1.4, 0.2]})
    assert len(pred.predictions) == 1

    meta = adapter.get_metadata()
    assert meta.framework == "sklearn"
    assert meta.n_features == 4

    adapter.unload()
