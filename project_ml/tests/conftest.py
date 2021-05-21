from typing import NoReturn, List, Tuple
from py._path.local import LocalPath

import pytest
import pandas as pd
from faker import Faker

from src.entities import FeatureParams, RandomForestClassifierParams, TrainPipelineParams, SplitParams, PathParams, PredictPipelineParams
from src.features.features import make_transformer, get_target_column
from src.pipelines.train_pipeline import train_pipeline


ROWS = 200


@pytest.fixture()
def output_predictions_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("test_predicitons.csv")


@pytest.fixture()
def load_model_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("test_model.pkl")


@pytest.fixture()
def metrics_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("test_metrics.json")


@pytest.fixture()
def load_transformer_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("test_transformer.pkl")


@pytest.fixture()
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture()
def target_col() -> str:
    return "target"


@pytest.fixture()
def normalize_numerical_true() -> bool:
    return True


@pytest.fixture()
def normalize_numerical_false() -> bool:
    return False


@pytest.fixture()
def feature_params_normalized(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        normalize_numerical_true: bool
) -> FeatureParams:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        normalize_numerical=normalize_numerical_true
    )
    return feature_params


@pytest.fixture()
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        normalize_numerical_false: bool
) -> FeatureParams:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        normalize_numerical=normalize_numerical_false
    )
    return feature_params


@pytest.fixture()
def fake_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(23)
    fake_data = {
        "age": [fake.pyint(min_value=30, max_value=80) for _ in range(ROWS)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(ROWS)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(ROWS)],
        "chol": [fake.pyint(min_value=126, max_value=555) for _ in range(ROWS)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(ROWS)],
        "thalach": [fake.pyint(min_value=71, max_value=202) for _ in range(ROWS)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(ROWS)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(ROWS)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(ROWS)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(ROWS)],
        "target": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)]
    }

    return pd.DataFrame(data=fake_data)


@pytest.fixture()
def fake_data_path(tmpdir: LocalPath, fake_data) -> LocalPath:
    data_path = tmpdir.join("test_data.csv")
    fake_data.to_csv(data_path)

    return data_path


@pytest.fixture()
def random_forest_training_params() -> RandomForestClassifierParams:
    model = RandomForestClassifierParams(
        model_type="RandomForestClassifier",
        n_estimators=20,
        max_depth=5,
        random_state=23
    )

    return model


@pytest.fixture()
def transformed_dataframe(
        fake_data: pd.DataFrame, feature_params_normalized: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    transformer = make_transformer(feature_params_normalized)
    transformer.fit(fake_data)

    transformed_features = transformer.transform(fake_data)
    target = get_target_column(fake_data, feature_params_normalized)

    return target, transformed_features


@pytest.fixture()
def train_pipeline_params(
    fake_data_path: str,
    load_model_path: str,
    metrics_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    normalize_numerical_true: bool,
    target_col: str,
    load_transformer_path: str,
    random_forest_training_params: RandomForestClassifierParams
) -> TrainPipelineParams:

    train_pipeline_parms = TrainPipelineParams(
        path_config=PathParams(
            input_data_path=fake_data_path,
            metrics_path=metrics_path,
            output_model_path=load_model_path,
            output_transformer_path=load_transformer_path,
        ),

        split_params=SplitParams(validation_size=0.2, random_state=21),

        feature_params=FeatureParams(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target_col=target_col,
            normalize_numerical=normalize_numerical_true
        ),

        train_params=random_forest_training_params
    )
    return train_pipeline_parms


@pytest.fixture()
def predict_pipeline_params(
    fake_data_path: str,
    load_model_path: str,
    output_predictions_path: str,
    load_transformer_path: str,
) -> PredictPipelineParams:

    pred_pipeline_params = PredictPipelineParams(
        input_data_path=fake_data_path,
        output_data_path=output_predictions_path,
        transformer_path=load_transformer_path,
        model_path=load_model_path,
    )
    return pred_pipeline_params


@pytest.fixture()
def train_on_fake_data(train_pipeline_params: TrainPipelineParams) -> NoReturn:
    train_pipeline(train_pipeline_params)
