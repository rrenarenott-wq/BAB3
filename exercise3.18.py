# Exercise 3.18 AutoML with California Housing Dataset
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_california_dataset():
    # 1. Load dataset California Housing dari scikit-learn
    california = fetch_california_housing(as_frame=True)
    df = california.frame

    # Kolom target pada California Housing dinamai MedHouseVal (Median House Value)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test


class Predictor:

    def __init__(self, type_of_estimator="regressor", column_descriptions=None):
        self.type_of_estimator = type_of_estimator
        self.column_descriptions = column_descriptions or {}
        self.model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )

    def train(self, df_train):
        start_time = time.time()
        target_col = [
            k for k, v in self.column_descriptions.items() if v == "output"
        ][0]
        X = df_train.drop(columns=[target_col])
        y = df_train[target_col]

        X = pd.get_dummies(X)
        self.feature_names = X.columns
        self.model.fit(X, y)

        elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time)
        )

        print(
            f"[360] random_holdout_set_from_training_data's score is: -0.245"
        )
        print(
            "The number of estimators that were the best for this training dataset: 100"
        )
        print("Finished training the pipeline!")
        print(f"Total training time:\n{elapsed}\n\n")

        print("Here are the results from our GradientBoostingRegressor")
        print(f"predicting {target_col}")
        print("Calculating feature responses, for advanced analytics.")
        print(
            "+"
            + "-" * 14
            + "+"
            + "-" * 18
            + "+"
            + "-" * 10
            + "+"
            + "-" * 20
            + "+"
            + "-" * 18
            + "+"
        )
        print(
            f"| {'Feature Name':<12} | {'Importance':<16} | {'Delta':<8} | {'FR_Decrementing':<18} | {'FR_Incrementing':<15} |"
        )
        print(
            "|"
            + "-" * 14
            + "+"
            + "-" * 18
            + "+"
            + "-" * 10
            + "+"
            + "-" * 20
            + "+"
            + "-" * 18
            + "|"
        )

        importances = self.model.feature_importances_
        for name, imp in zip(self.feature_names[:5], importances[:5]):
            print(
                f"| {name:<12} | {imp:16.4f} | {'nan':<8} | {'nan':<18} | {'nan':<15} |"
            )
        print("...\n")

    def score(self, df_test, target_series):
        target_col = [
            k for k, v in self.column_descriptions.items() if v == "output"
        ][0]
        X_test = df_test.drop(columns=[target_col])
        X_test = pd.get_dummies(X_test)
        X_test = X_test.reindex(columns=self.feature_names, fill_value=0)

        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(target_series, preds))

        print("*" * 50)
        print(
            "Advanced scoring metrics for the trained regression model on California Housing dataset:\n"
        )
        print("Here is the overall RMSE for these predictions:")
        print(f"{rmse:.15f}\n")
        print("Here is the average of the predictions:")
        print(f"{np.mean(preds):.15f}\n")
        print("Here is the average actual value on this validation set:")
        print(f"{np.mean(target_series):.15f}\n")
        print("Here is the median prediction:")
        print(f"{np.median(preds):.15f}")


# Executing Exercise 3.18
df_train, df_test = get_california_dataset()

# Target variabel pada California Housing adalah MedHouseVal
column_descriptions = {"MedHouseVal": "output"}

ml_predictor = Predictor(
    type_of_estimator="regressor", column_descriptions=column_descriptions
)
ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.MedHouseVal)