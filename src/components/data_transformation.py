import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_features(self, df, family_categories=None):
        try:
            logging.info("Starting feature engineering")

            # sort for time series (string family so order matches raw data)
            df = df.sort_values(["store_nbr", "family", "date"])

            # Fixed category order: same codes at train and predict (cat.codes alone shifts
            # if a different subset of families appears in the batch).
            if family_categories is None:
                family_categories = sorted(df["family"].unique().tolist())
            family_cat = pd.Categorical(df["family"], categories=family_categories)
            codes = family_cat.codes
            if (codes < 0).any():
                n_bad = int((codes < 0).sum())
                logging.warning(
                    "%d rows have family not in training categories; mapping those to 0",
                    n_bad,
                )
                codes = np.where(codes < 0, 0, codes)
            df["family"] = codes.astype(np.int32)

            # time features
            df["day_of_week"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month

            # promotion feature
            df["is_promo"] = (df["onpromotion"] > 0).astype(int)

            # lag features
            group = df.groupby(["store_nbr", "family"])["sales"]

            df["lag7"] = group.shift(7)
            df["lag14"] = group.shift(14)
            df["lag28"] = group.shift(28)

            # rolling mean
            df["rmean7"] = group.shift(7).rolling(7).mean()
            df["rmean14"] = group.shift(7).rolling(14).mean()

            # rolling std
            df["rstd7"] = group.shift(7).rolling(7).std()

            # drop NaN
            df = df.dropna()

            logging.info("Feature engineering completed")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, parse_dates=["date"])
            test_df = pd.read_csv(test_path, parse_dates=["date"])

            logging.info("Train and test data loaded")

            # combine for consistent feature creation
            df = pd.concat([train_df, test_df], axis=0)

            family_categories = sorted(df["family"].unique().tolist())
            df = self.create_features(df, family_categories=family_categories)

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                {"family_categories": family_categories},
            )

            # split back (copy so assignments are not on a view of df)
            train_df = df.loc[df["date"] < "2017-07-01"].copy()
            test_df = df.loc[df["date"] >= "2017-07-01"].copy()

            # target
            train_df["sales_log"] = np.log1p(train_df["sales"])

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

            X_train = train_df[FEATURES]
            y_train = train_df["sales_log"]

            X_test = test_df[FEATURES]
            y_test = test_df["sales"]

            # convert to array
            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            logging.info("Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)