from dataclasses import dataclass
from .split_params import SplitParams
from .feature_params import FeatureParams
from .train_params import RandomForestClassifierParams
from .path_params import PathParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainPipelineParams:
    """ Defines train pipeline parameters. """
    path_config: PathParams
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: RandomForestClassifierParams


TrainPipelineParamsSchema = class_schema(TrainPipelineParams) 