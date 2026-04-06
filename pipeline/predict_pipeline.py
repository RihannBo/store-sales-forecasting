import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data_path):
        try:
            logging.info("Loading input data for prediction")

            df = pd.read_csv(data_path, parse_dates=["date"])

            transformer = DataTransformation()
            prep = load_object(transformer.data_transformation_config.preprocessor_obj_file_path)
            family_categories = prep["family_categories"]

            # Same create_features + same fixed family vocabulary as training
            df = transformer.create_features(df, family_categories=family_categories)

            FEATURES = [
                "store_nbr",
                "family",
                "onpromotion",
                "is_promo",
                "day_of_week",
                "month",
                "lag7", "lag14", "lag28",
                "rmean7", "rmean14",
                "rstd7",
            ]

            X = df[FEATURES].to_numpy()

            logging.info("Loading trained model")
            model = load_object("artifacts/model.pkl")

            y_pred_log = model.predict(X)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.clip(y_pred, 0, None)

            logging.info("Prediction completed")

            return y_pred

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = PredictPipeline()

    preds = pipeline.predict("data/train.csv")

    print("\n🔹 Sample predictions:")
    print(preds[:10])

    print("\n🔹 Random predictions:")
    idx = np.random.choice(len(preds), 10)
    print(preds[idx])

    print("\n🔹 Prediction stats:")
    print(f"Min   : {preds.min():.2f}")
    print(f"Max   : {preds.max():.2f}")
    print(f"Mean  : {preds.mean():.2f}")
    print(f"Median: {np.median(preds):.2f}")