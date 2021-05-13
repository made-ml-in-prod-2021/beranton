from typing import NoReturn

import pandas as pd

from src.data import split_train_val
from src.entities import SplitParams


def test_split_train_val_data(fake_data: pd.DataFrame) -> NoReturn:

    test_size = 0.2
    params = SplitParams(validation_size=test_size, random_state=21)
    train_df, test_df = split_train_val(fake_data, params)

    assert len(train_df) >= len(fake_data) * 0.2
    assert len(test_df) <= len(fake_data) * 0.2

    assert len(train_df) + len(test_df) == len(fake_data)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
