from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    """ Defines feature types. """

    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    normalize_numerical: bool 