from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def predict(model: RandomForestClassifier, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)

    return predictions


def evaluate(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "Accuracy": accuracy_score(target, predictions),
        "F1": f1_score(target, predictions)
    }
