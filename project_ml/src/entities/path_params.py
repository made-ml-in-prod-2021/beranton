from dataclasses import dataclass


@dataclass()
class PathParams:
    """ Defines data and models paths."""

    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metrics_path: str
