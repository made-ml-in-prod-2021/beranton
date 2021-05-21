import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from .transformers import Transformer
from src.entities.feature_params import FeatureParams


def preprocess_numerical_features_normalized() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("custom_transformer", Transformer()),
        ]
    )

    return numerical_pipeline


def preprocess_numerical_features() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))
        ]
    )

    return numerical_pipeline


def preprocess_categorical_features() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("one_hot", OneHotEncoder(drop="if_binary")),
        ]
    )

    return categorical_pipeline


def make_transformer(params: FeatureParams) -> ColumnTransformer:
    if params.normalize_numerical:
        transformer = ColumnTransformer(
            [
                ("categorical_pipeline",
                 preprocess_categorical_features(),
                 params.categorical_features),

                ("numerical_pipeline",
                 preprocess_numerical_features_normalized(),
                 params.numerical_features)
            ]
        )

    else:
        transformer = ColumnTransformer(
            [
                ("categorical_pipeline",
                 preprocess_categorical_features(),
                 params.categorical_features),
                ("numerical_pipeline",
                 preprocess_numerical_features(),
                 params.numerical_features),
            ]
        )

    return transformer


def get_target_column(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
