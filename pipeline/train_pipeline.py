import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    try:
        logging.info("Training pipeline started")

        # 1. Data ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed")

        # 2. Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

        logging.info("Data transformation completed")

        # 3. Model training
        model_trainer = ModelTrainer()
        rmsle = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model training completed. Validation RMSLE: {rmsle}")
        print(f"Validation RMSLE: {rmsle}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()