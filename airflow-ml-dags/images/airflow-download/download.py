import os

import click
from sklearn.datasets import load_breast_cancer


@click.command()
@click.option("--output-dir", required=True)
@click.option("--output-dir-features", required=True)
@click.option("--output-dir-target", required=True)
def download(output_dir: str, output_dir_features: str, output_dir_target: str):

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(output_dir_features, index=False)
    y.to_csv(output_dir_target, index=False)


if __name__ == '__main__':
    download()