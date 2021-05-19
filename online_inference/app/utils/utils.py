import pickle
from typing import NoReturn


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


from .transformers import Transformer


def load_pkl(input_file: str):
    with open(input_file, "rb") as fin:
        data = pickle.load(fin)

    return data


def load_model(input_file: str) -> RandomForestClassifier:
    model = load_pkl(input_file)

    return model


def load_transformer(input_file: str) -> Transformer:
    transformer = load_pkl(input_file)

    return transformer
