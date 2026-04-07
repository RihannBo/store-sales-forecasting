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
            logging.info("Loading history and new data")

            history_path = "artifacts/history.csv"

            # load history (has sales)
            history_df = pd.read_csv(
                history_path,
                parse_dates=["date"],
                dtype={"store_nbr": "int32", "family": "string"},
                low_memory=False,
            )
            history_df = history_df.copy()

            # load new data (NO sales)
            new_df = pd.read_csv(data_path, parse_dates=["date"])
            new_df["__is_new"] = 1
            history_df["__is_new"] = 0

            # combine
            df = pd.concat([history_df, new_df], axis=0)
            df = df.sort_values(["store_nbr", "family", "date"])

            transformer = DataTransformation()
            prep = load_object(
                transformer.data_transformation_config.preprocessor_obj_file_path
            )
            family_categories = prep["family_categories"]

            df = transformer.create_features(
                df, family_categories=family_categories
            )

            # select only NEW rows (test data) using explicit marker
            df_new = df.loc[df["__is_new"] == 1].copy()

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

            if df_new.empty:
                raise ValueError(
                    "No prediction rows left after feature creation. "
                    "Need enough history per (store_nbr, family) to build lag features."
                )
            logging.info("Number of prediction rows: %d", len(df_new))

            X = df_new[FEATURES].to_numpy()

            logging.info("Loading trained model")
            model = load_object("artifacts/model.pkl")

            y_pred_log = model.predict(X)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.clip(y_pred, 0, None)

            # update history with predictions
            df_new.loc[:, "sales"] = y_pred
            updated_history = pd.concat([history_df, df_new], axis=0)
            updated_history = updated_history.drop(columns=["__is_new"], errors="ignore")

            updated_history.to_csv(history_path, index=False)

            logging.info("Prediction completed")

            return y_pred

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = PredictPipeline()

    preds = pipeline.predict("data/test.csv")

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