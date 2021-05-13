from typing import NoReturn

import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.features import get_target_column, make_transformer

from src.entities import FeatureParams
from src.features import Transformer


def test_extract_target(fake_data: pd.DataFrame, feature_params_normalized: FeatureParams) -> NoReturn:
    target_df = get_target_column(fake_data, feature_params_normalized)

    assert len(target_df) == len(fake_data)
    assert fake_data[feature_params_normalized.target_col].equals(target_df)


def test_custom_transformer(fake_data: pd.DataFrame, feature_params_normalized: FeatureParams) -> NoReturn:
    fake_data_np = fake_data[feature_params_normalized.numerical_features].to_numpy()
    correct_fake_np = (fake_data_np - fake_data_np.mean(axis=0)) / fake_data_np.std(axis=0)

    transformer = Transformer()
    transformer.fit(fake_data_np)

    custom_scaled_data = transformer.transform(fake_data_np)

    assert custom_scaled_data.shape == correct_fake_np.shape
    assert np.allclose(custom_scaled_data, correct_fake_np)


def test_build_features_pipeline_normalized(
        fake_data: pd.DataFrame, feature_params_normalized: FeatureParams
) -> NoReturn:
    transformer = make_transformer(feature_params_normalized)

    transformer.fit(fake_data)
    check_is_fitted(transformer)

    transformed_data = transformer.transform(fake_data)

    synth_data_np = fake_data[feature_params_normalized.numerical_features].to_numpy()
    correct_synth_np = (synth_data_np - synth_data_np.mean(axis=0)) / synth_data_np.std(axis=0)

    num_features = len(feature_params_normalized.numerical_features)
    transformed_cols = transformed_data[:, -num_features:]

    assert np.allclose(transformed_cols, correct_synth_np)
    assert not pd.isnull(transformed_data).any().any()
    assert (fake_data.shape[0], 27) == transformed_data.shape


def test_build_features_pipeline(
        fake_data: pd.DataFrame, feature_params: FeatureParams
) -> NoReturn:
    transformer = make_transformer(feature_params)

    transformer.fit(fake_data)
    check_is_fitted(transformer)

    transformed_data = transformer.transform(fake_data)

    synth_data_np = fake_data[feature_params.numerical_features].to_numpy()

    num_features = len(feature_params.numerical_features)
    transformed_cols = transformed_data[:, -num_features:]

    assert np.allclose(transformed_cols, synth_data_np)
    assert not pd.isnull(transformed_data).any().any()
    assert (fake_data.shape[0], 27) == transformed_data.shape
