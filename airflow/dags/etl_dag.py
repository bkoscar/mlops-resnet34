import sys 
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

sys.path.append("/opt/airflow/src")
from extract_data import KaggleDatasetDownloader, TransformedData, MetadataCreate, DatasetSplitter

CONFIG_DIR = "/opt/airflow/config/.kaggle"
DATA_RAW = "/opt/airflow/data/data_raw"
DATA_TRAINABLE = "/opt/airflow/data/data_trainable"
METADATA_PATH = "/opt/airflow/data/metadata.csv"
OUTPUT_METADATA_PATH = "/opt/airflow/data/metadata_split.csv"

def download_dataset():
    downloader = KaggleDatasetDownloader(config_dir=CONFIG_DIR, download_path=DATA_RAW)
    downloader.download_dataset("andrewmvd/animal-faces")

def transform_data():
    transformer = TransformedData(source_path=f"{DATA_RAW}/afhq", dest_folder=DATA_TRAINABLE)
    transformer.copy_data()

def create_metadata():
    metadata_creator = MetadataCreate(dataset_folder_path=DATA_TRAINABLE, metadata_path=METADATA_PATH)
    metadata_creator.create_metadata()

def split_dataset():
    splitter = DatasetSplitter(metadata_path=METADATA_PATH, output_metadata_path=OUTPUT_METADATA_PATH)
    splitter.split_dataset(train_split=0.8, test_split=0.5)

# Define the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    description="Pipeline DAG to download, transform, create metadata, and split a dataset",
    schedule_interval=None,  # Run manually
    start_date=datetime(2025, 3, 29),
    catchup=False,
) as dag:

    # DAG tasks
    download_task = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    create_metadata_task = PythonOperator(
        task_id="create_metadata",
        python_callable=create_metadata,
    )

    split_dataset_task = PythonOperator(
        task_id="split_dataset",
        python_callable=split_dataset,
    )

    download_task >> transform_task >> create_metadata_task >> split_dataset_task
