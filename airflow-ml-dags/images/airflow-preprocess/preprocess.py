import os
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):

    X = pd.read_csv(os.path.join(input_dir, "features.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))

    y.set_axis(["target"], axis=1)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    data_preprocessed = pd.concat([X, y], axis=1)
    data_preprocessed.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)


if __name__ == "__main__":
    preprocess()
