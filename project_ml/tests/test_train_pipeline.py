import os

from src.entities import TrainPipelineParams
from src.train_pipeline import train_pipeline


def test_train_pipeline(
    train_pipeline_params: TrainPipelineParams,
    metrics_path: str,
    load_model_path: str,
    load_transformer_path: str
):

    metrics = train_pipeline(train_pipeline_params)
    assert 0 < metrics["ROC_AUC"] <= 1
    assert 0 < metrics["Accuracy"] <= 1
    assert 0 < metrics["F1"] <= 1

    assert os.path.exists(load_transformer_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(load_model_path)