import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_rmsle


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process")

            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")

            # Model (column 1 = family: integer codes but unordered categorical)
            cat_mask = np.zeros(X_train.shape[1], dtype=bool)
            cat_mask[1] = True

            model = HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
                categorical_features=cat_mask,
            )

            logging.info("Fitting model")
            model.fit(X_train, y_train)

            # Predictions (log → original scale)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.clip(y_pred, 0, None)

            # Evaluate (RMSLE)
            rmsle = evaluate_rmsle(y_test, y_pred)

            logging.info(f"Model evaluation completed. RMSLE: {rmsle}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model saved successfully")

            return rmsle

        except Exception as e:
            raise CustomException(e, sys)