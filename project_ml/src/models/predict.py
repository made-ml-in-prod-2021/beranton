from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def predict(model: RandomForestClassifier, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)

    return predictions

def evaluate(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "ROC_AUC": roc_auc_score(target, predictions),
        "Accuracy": accuracy_score(target, predictions),
        "F1": f1_score(target, predictions)
    }