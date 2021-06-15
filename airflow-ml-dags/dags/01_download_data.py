from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR, \
    DATA_RAW_FEATURES_FILE_NAME, DATA_RAW_TARGET_FILE_NAME


with DAG(
    "data_download",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0, 2),
) as dag:

    features_path = DATA_RAW_DIR + '/' + DATA_RAW_FEATURES_FILE_NAME
    target_path = DATA_RAW_DIR + '/' + DATA_RAW_TARGET_FILE_NAME

    data_download = DockerOperator(
        image="airflow-download",
        command=f"--output-dir {DATA_RAW_DIR} --output-dir-features {features_path} --output-dir-target {target_path}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )