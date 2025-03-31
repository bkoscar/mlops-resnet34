import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


class KaggleDatasetDownloader:
    def __init__(self, config_dir, download_path):
        """
        Class for downloading datasets from Kaggle.

        Args:
            config_dir (str): Path to the directory where the kaggle.json file is located.
            download_path (str): Path where downloaded data will be saved.
        """
        os.environ["KAGGLE_CONFIG_DIR"] = config_dir
        self.download_path = download_path
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self, dataset_name):
        """
        Downloads a dataset from Kaggle and unzips it.

        Args:
            dataset_name (str): Dataset name on Kaggle (e.g., "andrewmvd/animal-faces").
        """
        os.makedirs(self.download_path, exist_ok=True)
        self.api.dataset_download_files(
            dataset_name,
            path=self.download_path,
            unzip=True
        )
        print(f"Dataset downloaded to: {self.download_path}")


class TransformedData:
    def __init__(self, source_path, dest_folder):
        """
        Class for transforming and copying data.

        Args:
            source_path (str): Path to the source folder.
            dest_folder (str): Path to the destination folder.
        """
        self.source_path = source_path
        self.dest_folder = dest_folder

    def copy_data(self):
        """
        Copies image files from the source folder to the destination folder,
        preserving the 'train' and 'val' subfolders.
        """
        subfolders = ["train", "val"]
        os.makedirs(self.dest_folder, exist_ok=True)
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.source_path, subfolder)
            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                for root, _, files in os.walk(subfolder_path):
                    for filename in files:
                        if filename.endswith(".jpg"):
                            src = os.path.join(root, filename)
                            dst = os.path.join(self.dest_folder, filename)
                            shutil.copy(src, dst)
                            print(f"Copied: {src} → {dst}")


class MetadataCreate:
    def __init__(self, dataset_folder_path, metadata_path):
        """
        Class for creating a metadata file.

        Args:
            dataset_folder_path (str): Path to the folder with processed data.
            metadata_path (str): Path where the metadata.csv file will be saved.
        """
        self.dataset_folder_path = dataset_folder_path
        self.metadata_path = metadata_path

    def create_metadata(self):
        """
        Creates a CSV file with metadata for the images in the destination folder.
        """
        all_files = [
            file
            for file in os.listdir(self.dataset_folder_path)
            if file.endswith(".jpg")
            and os.path.isfile(os.path.join(self.dataset_folder_path, file))
        ]
        df = pd.DataFrame(all_files, columns=["file_name"])
        df["source_label"] = df["file_name"].apply(lambda x: x.split("_")[1])
        df.to_csv(self.metadata_path, index=False)
        print(f"Metadata created at: {self.metadata_path}")


class DatasetSplitter:
    def __init__(self, metadata_path, output_metadata_path):
        """
        Class for splitting the dataset into train, val, and test.

        Args:
            metadata_path (str): Path to the metadata.csv file.
            output_metadata_path (str): Path where the metadata file with splits will be saved.
        """
        self.metadata_path = metadata_path
        self.output_metadata_path = output_metadata_path

    def split_dataset(self, train_split=0.8, test_split=0.5):
        """
        Splits the dataset into train, val, and test, and updates the metadata file.

        Args:
            train_split (float): Proportion of data to use for training.
            test_split (float): Proportion of the remaining data to use for testing (vs. validation).
        """
        metadata = pd.read_csv(self.metadata_path)
        X = metadata["file_name"]
        y = metadata["source_label"]

        # Split into train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - train_split, random_state=42, stratify=y
        )

        # Split temp into val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_split, random_state=42, stratify=y_temp
        )

        # Add the "split" column to metadata
        metadata["split"] = "unknown"
        metadata.loc[X_train.index, "split"] = "train"
        metadata.loc[X_val.index, "split"] = "val"
        metadata.loc[X_test.index, "split"] = "test"

        # Save the updated metadata file
        metadata.to_csv(self.output_metadata_path, index=False)
        print(f"Metadata with splits saved at: {self.output_metadata_path}")


def test_pipeline():
    """
    Function that runs the complete flow of downloading, transforming,
    creating metadata, and splitting the dataset.
    """
    # Path configuration
    config_dir = "/opt/airflow/config/.kaggle"
    data_raw = "/opt/airflow/data/data_raw"
    data_trainable = "/opt/airflow/data/data_trainable"
    metadata_path = "/opt/airflow/data/metadata.csv"
    output_metadata_path = "/opt/airflow/data/metadata_split.csv"

    # Download dataset
    downloader = KaggleDatasetDownloader(config_dir=config_dir, download_path=data_raw)
    downloader.download_dataset("andrewmvd/animal-faces")

    # Transform data
    transformer = TransformedData(source_path=f"{data_raw}/afhq", dest_folder=data_trainable)
    transformer.copy_data()

    # Create metadata
    metadata_creator = MetadataCreate(dataset_folder_path=data_trainable, metadata_path=metadata_path)
    metadata_creator.create_metadata()

    # Split dataset
    splitter = DatasetSplitter(metadata_path=metadata_path, output_metadata_path=output_metadata_path)
    splitter.split_dataset(train_split=0.8, test_split=0.5)


if __name__ == "__main__":
    test_pipeline()
