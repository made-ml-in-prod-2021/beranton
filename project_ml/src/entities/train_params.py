from dataclasses import dataclass, field


@dataclass()
class RandomForestClassifierParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default="100")
    max_depth: int = field(default="10")
    random_state: int = field(default="42") 