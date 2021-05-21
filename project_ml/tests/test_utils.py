import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import read_data, load_model


def test_read_data(fake_data_path: str):

    data = read_data(fake_data_path)

    assert isinstance(data, pd.DataFrame)
    assert (200, 15) == data.shape
