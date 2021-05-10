import os
import logging.config

import pandas as pd
from omegaconf import DictConfig
import hydra

from src.entities.predict_pipeline_params import PredictPipelineParams, PredictPipelineParamsSchema
from src.models import predict
from src.utils import read_data, load_pkl

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info("Start prediction")
    logger.info("Load data")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info("Finished loading data")

    logger.info("Load pretrained transformer")
    transformer = load_pkl(predict_pipeline_params.pipeline_path)
    logger.info("Finished loading pretrained transformer")
    transformed_data = pd.DataFrame(transformer.transform(data))

    logger.info("Load pretrained model")
    model = load_pkl(predict_pipeline_params.model_path)
    logger.info("Finished loading pretrained model")

    logger.info("Make predictions")
    predictions = predict(model, transformed_data)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(predict_pipeline_params.output_data_path, header=False)
    logger.info(f"Prediction saved to file{predict_pipeline_params.output_data_path}")

    return predictions


@hydra.main(config_path="../configs", config_name="predict_config")
def start_predict_pipeline(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = PredictPipelineParamsSchema()
    params = schema.load(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    start_predict_pipeline()