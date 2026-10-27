"""Generates one synthetic model per framework, checked in as small
fixture files under tests/test_adapters/fixtures/. Run once:
    python tests/test_adapters/make_fixtures.py
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def make_sklearn_fixture():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    clf.fit(X, y)
    path = os.path.join(FIXTURES_DIR, "iris_rf.pkl")
    joblib.dump(clf, path)
    return path


def make_onnx_fixture():
    """Builds a small linear-regression ONNX graph directly with onnx.helper
    (avoids a skl2onnx <-> onnx version pin conflict for this MVP fixture --
    the point is exercising the ONNX adapter code path, not model fidelity)."""
    import onnx
    from onnx import helper, TensorProto

    n_features = 4
    W = np.array([[0.5, -0.3, 0.8, 0.1]], dtype=np.float32).T  # (4, 1)
    b = np.array([0.05], dtype=np.float32)

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.tolist())

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "b"], ["output"])

    graph = helper.make_graph(
        [matmul_node, add_node],
        "iris_linear",
        [input_tensor],
        [output_tensor],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, producer_name="modelmint-fixtures")
    model.opset_import[0].version = 13
    model.ir_version = 8  # keep compatible with the pinned onnxruntime's max supported IR version
    onnx.checker.check_model(model)

    path = os.path.join(FIXTURES_DIR, "iris_rf.onnx")
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    p1 = make_sklearn_fixture()
    p2 = make_onnx_fixture()
    print("wrote", p1)
    print("wrote", p2)
