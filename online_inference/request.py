import os

import click
import requests

import pandas as pd
import numpy as np


SERVICE_HOST = os.environ.get("HOST", default="0.0.0.0")
SERVICE_PORT = os.environ.get("PORT", default="8080")


def load_data(filepath: str):
    data = pd.read_csv(filepath)

    return data


@click.comman()
@click.option("--file_path", default="data.csv")
@click.option("--count", deafult=1, help="Number of data rows")
def make_request(file_path: str, count: int):
    data = load_data(file_path)
    features = list(data.columns)

    if count > len(data):
        count = len(data)

    for i in range(count):
        request_data = [x.item() if isinstance(x, np.generic) else x for in data.iloc[i].tolist()]
        request_data = dict(zip(features, request_data))
        response = requests.get(
            url=f"http://{SERVICE_HOST}:{SERVICE_PORT}/predict",
            json=request_data
        )
        if response.status_code == 200:
            click.echo(f"Features data\t {request_data}")
            click.echo(f"Label:\t {response.json()}")
        else:
            click.echo(f"Failed to make prediction, ERROR {response.status_code}: {response.text}")


if __name__ == "__main__":
    make_request()