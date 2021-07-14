import pickle
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--input-dir")
@click.option("--input-model-dir")
@click.option("--output-dir")
@click.option("--model-name")
@click.option("--data-file-name")
def predict(
    input_dir: str,
    input_model_dir: str,
    output_dir: str,
    model_name: str,
    data_file_name: str,
):

    input_data_path = Path(input_dir)
    input_model_dir = Path(input_model_dir)
    output_dir_path = Path(output_dir)

    with open(input_model_dir / model_name, "rb") as fin:
        model = pickle.load(fin)

    data = pd.read_csv(input_data_path / data_file_name)

    predictions = model.predict(data)
    df_predictions = pd.DataFrame(predictions, columns="target")

    output_dir_path.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(output_dir_path, index=False)


if __name__ == "__main__":
    predict()
