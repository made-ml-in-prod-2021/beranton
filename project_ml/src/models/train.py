import pandas as pd
from sklearn.ensemble import RandomForestClassifier


from src.entities.train_params import RandomForestClassifierParams


def train(features: pd.DataFrame, target: pd.Series,
          train_params: RandomForestClassifierParams) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=train_params.n_estimators,
        max_depth=train_params.max_depth,
        random_state=train_params.random_state
    )

    model.fit(features, target)

    return model
