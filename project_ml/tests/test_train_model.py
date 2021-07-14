from typing import NoReturn, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.entities import RandomForestClassifierParams
from src.models import train


def test_random_forest_train_model(
    random_forest_training_params: RandomForestClassifierParams,
    transformed_dataframe: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_dataset = transformed_dataframe
    model = train(transformed_dataset, target, random_forest_training_params)

    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)