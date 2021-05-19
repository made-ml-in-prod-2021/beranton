import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Transformer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.mean) / self.std

        return X
