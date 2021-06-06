import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import read_data, load_pkl


def test_read_data(fake_data_path: str):

    data = read_data(fake_data_path)

    assert isinstance(data, pd.DataFrame)
    assert (200, 15) == data.shape


def test_load_pkl(load_model_path: str):
    model = load_pkl(load_model_path)
    assert isinstance(model, RandomForestClassifier)