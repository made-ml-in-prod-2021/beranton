from typing import Tuple


import pandas as pd
from sklearn.model_selection import train_test_split


from src.entities import SplitParams


def split_train_val(
    data: pd.DataFrame,
    params: SplitParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, val_data = train_test_split(
        data, test_size=params.validation_size, random_state=params.random_state
    )

    return train_data, val_data