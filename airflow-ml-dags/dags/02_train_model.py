from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constant import (
    DEFAULT_ARGS,
    DATA_RAW_DIR,
    DATA_VOLUME_DIR,
    DATA_RAW_FEATURES_FILE_NAME,
    DATA_RAW_TARGET_FILE_NAME,
    DATA_PROCESSED_DIR,
    MODELS_DIR_DEFAULT,
)


with DAG(
    "train",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=days_ago(2),
) as dag:

    features_path = DATA_RAW_DIR + "/" + DATA_RAW_FEATURES_FILE_NAME
    target_path = DATA_RAW_DIR + "/" + DATA_RAW_TARGET_FILE_NAME

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {DATA_RAW_DIR} --output-dir {DATA_PROCESSED_DIR} ",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {DATA_PROCESSED_DIR}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {DATA_PROCESSED_DIR} --output-dir {MODELS_DIR_DEFAULT}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {DATA_PROCESSED_DIR} --input-model-dir {MODELS_DIR_DEFAULT}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    preprocess >> split >> train >> validate
