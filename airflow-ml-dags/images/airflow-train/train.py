import os
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):

    input_data_path = Path(input_dir)
    data = pd.read_csv(input_data_path / "train.csv")

    y = data[["target"]]
    X = data.drop(["target"], axis=1)

    model = LogisticRegression()
    model.fit(X, y)

    output_model_path = Path(output_dir)
    output_model_path.mkdir(exist_ok=True, parents=True)

    current_output_model_path = os.path.join(output_model_path, "model")
    with open(current_output_model_path, "wb") as fout:
        pickle.dump(model, fout)


if __name__ == "main":
    train()
