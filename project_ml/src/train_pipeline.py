import os
import logging.config
from typing import Dict

from omegaconf import DictConfig
import pandas as pd
import hydra

from src.data import split_train_val
from src.entities.train_pipeline_params import TrainPipelineParams, TrainPipelineParamsSchema
from src.features.features import get_target_column, make_transformer
from src.models import train, predict, evaluate
from src.utils import read_data, save_pkl, save_metrics_json


logger = logging.getLogger("ml_project/train_pipeline")


def train_pipeline(training_pipeline_params: TrainPipelineParams) -> Dict[str, float]:

    logger.info(f"Train pipeline parameters {training_pipeline_params}")
    logger.info(f"Model type: {training_pipeline_params.train_params.model_type}")

    logger.info("Load data")
    data = read_data(training_pipeline_params.path_config.input_data_path)
    train_X, val_X = split_train_val(data, training_pipeline_params.split_params)
    logger.info("Finished loading data")

    logger.info("Preprocess data")
    transformer = make_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_X)
    save_pkl(transformer, training_pipeline_params.path_config.output_transformer_path)
    train_features = pd.DataFrame(transformer.transform(train_X))
    train_target = get_target_column(train_X, training_pipeline_params.feature_params)
    logger.info("Finished preprocessing data")

    logger.info("Train model")
    model = train(train_features, train_target, training_pipeline_params.train_params)
    logger.info("Finished training model")

    logger.info("Evaluate model")
    val_features = pd.DataFrame(transformer.transform(val_X))
    val_target = get_target_column(val_X, training_pipeline_params.feature_params)
    predictions = predict(model, val_features)
    metrics = evaluate(predictions, val_target)
    
    save_metrics_json(training_pipeline_params.path_config.metrics_path, metrics)
    logger.info("Finished evaluating model")
    logger.info(f"Model scores: {metrics}")

    save_pkl(model, training_pipeline_params.path_config.output_model_path)
    logger.info("Save model")

    return metrics


@hydra.main(config_path="../configs", config_name="train_config")
def start_train_pipeline(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    start_train_pipeline()