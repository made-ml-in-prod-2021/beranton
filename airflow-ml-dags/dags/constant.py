import datetime
from airflow.models import Variable


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["dummymail@acme.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

DATA_RAW_DIR = "/data/raw/{{ ds }}"
DATA_VOLUME_DIR = Variable.get("data_path")

DATA_RAW_FEATURES_FILE_NAME = "features.csv"
DATA_RAW_TARGET_FILE_NAME = "target.csv"

DATA_PROCESSED_DIR = "data/processed/{{ ds }}"

MODEL_FILE_NAME = "data/models"
MODELS_DIR_DEFAULT = "data/models"

MODEL_FILE_NAME_PREDICTION = "model"
MODELS_PATH_PREDICTION = "data/models"

DATA_PREDICTION_FILE_NAME = "features.csv"

DATA_PREDICTION_DIR = "data/predictions/{{ ds }}"
