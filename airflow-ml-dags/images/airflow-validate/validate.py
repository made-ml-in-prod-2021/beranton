import json
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


@click.command()
@click.option("--input-dir")
@click.option("--input-model-dir")
def validate(input_model_dir: str, input_dir: str):

    input_model_path = Path(input_model_dir)
    input_dataset_path = Path(input_dir)

    with open(input_model_path / "model", "rb") as fin:
        model = pickle.load(fin)

    data = pd.read_csv(input_dataset_path / "val.csv")
    y_val = data[["target"]]
    X_val = data.drop(["target"], axis=1)

    predictions = model.predict(X_val)

    metrics = {
        "Accuracy": accuracy_score(y_val.values, predictions),
        "ROC_AUC": roc_auc_score(y_val.values, predictions),
    }

    with open(input_model_path / "metrics.json", "w") as fout:
        json.dump(metrics, fout)


if __name__ == "__main__":
    validate()
