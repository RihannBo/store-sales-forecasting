import os
import sys
from dataclasses import dataclass
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")

        try:
            # Load your dataset
            df = pd.read_csv('data/train.csv', parse_dates=["date"])
            logging.info("Dataset loaded successfully")

            # sort for time series
            df = df.sort_values("date")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # TIME-BASED SPLIT (important)
            split_date = "2017-07-01"

            train_set = df[df["date"] < split_date]
            test_set = df[df["date"] >= split_date]

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Time-based train/test split completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    # transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))