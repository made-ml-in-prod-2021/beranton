import json
import pickle
from typing import NoReturn


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


from src.features.transformers import Transformer


def read_data(path: str) -> pd.DataFrame:

    data = pd.read_csv(path)

    return data


def save_metrics_json(file_path: str, metrics: dict) -> NoReturn:

    with open(file_path, "w") as metric_file:
        json.dump(metrics, metric_file)


def save_pkl(input_file, output_name: str) -> NoReturn:
    with open(output_name, "wb") as fout:
        pickle.dump(input_file, fout)


def load_pkl(input_file: str):
    with open(input_file, "rb") as fin:
        data = pickle.load(fin)

    return data


def save_model(model: RandomForestClassifier, output_name: str) -> NoReturn:
    save_pkl(model, output_name)


def save_transformer(transformer: Transformer, output_name: str) -> NoReturn:
    save_pkl(transformer, output_name)


def load_model(input_file: str) -> RandomForestClassifier:
    model = load_pkl(input_file)

    return model


def load_transformer(input_file: str) -> Transformer:
    transformer = load_pkl(input_file)

    return transformer


def save_predictions(predictions: pd.DataFrame, output_file_path: str, header=False) -> NoReturn:
    predictions.to_csv(output_file_path, header=header)
