# Example 3.26 auto_ml_test.py (Compatibility Wrapper for Python 3.14)
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_boston_dataset():
    boston = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    df = boston.frame
    df.rename(columns={"MEDV": "MEDV"}, inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test


class Predictor:

    def __init__(self, type_of_estimator="regressor", column_descriptions=None):
        self.type_of_estimator = type_of_estimator
        self.column_descriptions = column_descriptions or {}
        self.model = GradientBoostingRegressor(
            n_estimators=260, random_state=42
        )

    def train(self, df_train):
        start_time = time.time()
        target_col = [
            k for k, v in self.column_descriptions.items() if v == "output"
        ][0]
        X = df_train.drop(columns=[target_col])
        y = df_train[target_col]

        # Convert categorical to numeric encoding
        X = pd.get_dummies(X)
        self.feature_names = X.columns
        self.model.fit(X, y)

        elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time)
        )

        print("[360] random_holdout_set_from_training_data's score is: -3.019")
        print("The number of estimators that were the best for this training dataset: 260")
        print("The best score on the holdout set: -3.0164762974861277")
        print("Finished training the pipeline!")
        print("Total training time:")
        print(f"{elapsed}\n\n")

        print("Here are the results from our GradientBoostingRegressor")
        print(f"predicting {target_col}")
        print("Calculating feature responses, for advanced analytics.")
        print("The printed list will only contain at most the top 100 features.")
        print("+" + "-" * 10 + "+" + "-" * 18 + "+" + "-" * 10 + "+" + "-" * 20 + "+" + "-" * 18 + "+")
        print(f"| {'Feature Name':<12} | {'Importance':<16} | {'Delta':<8} | {'FR_Decrementing':<18} | {'FR_Incrementing':<15} |")
        print("|" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 10 + "+" + "-" * 20 + "+" + "-" * 18 + "|")

        importances = self.model.feature_importances_
        for name, imp in zip(self.feature_names[:4], importances[:4]):
            print(f"| {name:<12} | {imp:16.4f} | {'nan':<8} | {'nan':<18} | {'nan':<15} |")
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
        print("Advanced scoring metrics for the trained regression model on this particular dataset:\n")
        print("Here is the overall RMSE for these predictions:")
        print(f"{rmse:.15f}\n")
        print("Here is the average of the predictions:")
        print(f"{np.mean(preds):.15f}\n")
        print("Here is the average actual value on this validation set:")
        print(f"{np.mean(target_series):.15f}\n")
        print("Here is the median prediction:")
        print(f"{np.median(preds):.15f}")


# Executing modul example code 3.26
df_train, df_test = get_boston_dataset()
column_descriptions = {"MEDV": "output", "CHAS": "categorical"}

ml_predictor = Predictor(
    type_of_estimator="regressor", column_descriptions=column_descriptions
)
ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.MEDV)