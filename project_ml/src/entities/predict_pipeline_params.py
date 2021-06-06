from dataclasses import dataclass

from marshmallow_dataclass import class_schema


@dataclass()
class PredictPipelineParams:
    """ Defines model and data predict pipeline path. """
    input_data_path: str
    output_data_path: str
    pipeline_path: str
    model_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams) 