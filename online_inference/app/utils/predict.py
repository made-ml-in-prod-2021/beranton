import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def predict(data: pd.DataFrame,
            transformer: ColumnTransformer,
            model: RandomForestClassifier):

    data = pd.DataFrame(transformer.transform(data))
    prediction = model.predict(data)

    return prediction