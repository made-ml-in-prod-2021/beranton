import os

import pytest
from fastapi.testclient import TestClient

from prediction import *
from utils.prediction import predict
from utils.prediction import InputDataModel
from utils.utils import load_model, load_transformer

SRC_DIR = "src"


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_data():
    features = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 139,
        "chol": 200, "fbs": 0, "restecg": 1, "thalach": 172, 
        "exang": 0, "oldpeak": 2.1, "slope": 0, "ca": 0, "thal": 1
    }
    label = predict(
        data=InputDataModel(**features).convert_to_pandas(),
        transformer=load_model(os.path.join(SRC_DIR, "transformer.pkl")),
        model=load_transformer(os.path.join(SRC_DIR, "model.pkl")),
    )

    return features, label


def test_load_on_start(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() is True


@pytest.mark.parametrize(
    ["feature", "wrong_value", "expected_status_code"],
    [
        pytest.param("trestbps", 600, 400),
        pytest.param("sex", 6, 400),
        pytest.param("cp", 23, 400),
        pytest.param("age", 180, 400),
    ]
)
def test_predict_code_400(feature, wrong_value, expected_status_code, test_data, client):
    data = test_data[0].copy()
    data[feature] = wrong_value

    response = client.get("/predict", json=data)
    assert response.status_code == expected_status_code


def test_predict_labels(test_data, client):
    ground_truth = {"label": test_data[1]}
    response = client.get("/predict", json=test_data[0])

    assert response.status_code == 200
    assert response.json() == ground_truth