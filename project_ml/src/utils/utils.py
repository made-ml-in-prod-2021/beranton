import json
import pickle
from typing import NoReturn


import pandas as pd


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