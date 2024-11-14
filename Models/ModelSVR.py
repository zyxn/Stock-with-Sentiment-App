from Process.Dataproc import Dataproc, DatasetConfig
from typing import Dict
from dataclasses import dataclass
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import pickle
import matplotlib.pyplot as plt
import logging


@dataclass
class ModelEvaluation:
    rmse: float
    mse: float
    r2: float
    mae: float
    mape: float


class ModelSVR:
    def __init__(self, dataproc: Dataproc, params: Dict = None):
        self.dataproc = dataproc
        self.params = params
        self.return_type = ""
        self.X_train = self.dataproc.get_scaled_data().X_train_scaled
        self.X_val = self.dataproc.get_scaled_data().X_val_scaled
        self.y_train = None
        self.y_val = None
        self.model = SVR()
        self._set_params(params)
        self.evaluation = None
        logging.info("ModelSVR initialized with params: %s", params)

    def _set_params(self, params: Dict = None):
        if params is not None:
            self.model.set_params(**params)
            logging.info("Model parameters set to: %s", params)

    def create_model(self, use_returns: bool = False, return_type: str = ""):
        self.return_type = return_type
        logging.info(
            "Creating model with use_returns=%s and return_type='%s'",
            use_returns,
            return_type,
        )
        data_split = self.dataproc.get_data_split()
        scaled_data = self.dataproc.get_scaled_data()

        if use_returns:
            self._set_return_data(return_type)
        else:
            self.X_train, self.y_train = scaled_data.X_train_scaled, data_split.y_train
            self.X_val, self.y_val = scaled_data.X_val_scaled, data_split.y_val
            logging.info(
                "Data split without returns. Shapes - X_train: %s, y_train: %s",
                self.X_train.shape,
                self.y_train.shape,
            )

        self.model.fit(self.X_train, self.y_train)
        logging.info("Model training complete")
        evaluation_df = pd.DataFrame([self.evaluate().__dict__])
        self.evaluation = evaluation_df

    def _set_return_data(self, return_type: str):
        if return_type not in ["absolute", "relative", "log"]:
            logging.error("Invalid return type: %s", return_type)
            raise ValueError(
                "Invalid return type. Use 'absolute', 'relative', or 'log'."
            )

        logging.info("Setting return data with return_type='%s'", return_type)
        return_data_mapping = {
            "absolute": self.dataproc.get_returns()["y_train"].absolute_return,
            "relative": self.dataproc.get_returns()["y_train"].relative_return,
            "log": self.dataproc.get_returns()["y_train"].log_return,
        }

        return_data_mapping_val = {
            "absolute": self.dataproc.get_returns()["y_val"].absolute_return,
            "relative": self.dataproc.get_returns()["y_val"].relative_return,
            "log": self.dataproc.get_returns()["y_val"].log_return,
        }

        self.y_train = return_data_mapping[return_type]
        self.y_val = return_data_mapping_val[return_type]
        logging.info("Return data set successfully for training and validation")

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        logging.info("Making predictions for new data of shape: %s", new_data.shape)
        predictions = self.model.predict(new_data)
        logging.info("Predictions complete")
        return predictions

    def evaluate(self) -> ModelEvaluation:
        logging.info("Evaluating model")
        y_val_pred = self.model.predict(self.X_val)
        evaluation = ModelEvaluation(
            rmse=np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            mse=mean_squared_error(self.y_val, y_val_pred),
            r2=r2_score(self.y_val, y_val_pred),
            mae=mean_absolute_error(self.y_val, y_val_pred),
            mape=mean_absolute_percentage_error(self.y_val, y_val_pred),
        )
        logging.info("Evaluation complete: %s", evaluation)
        return evaluation

    def show_summary(self):
        target = "returns" if self.return_type else "close prices"
        logging.info("Showing summary for target: %s", target)
        print(f"Using {target} as target")
        print(self.model.get_params())
        evaluation_df = pd.DataFrame([self.evaluate().__dict__])
        print(evaluation_df)

    def plot_prediction(self):
        logging.info("Plotting predictions")
        y_val_pred = self.model.predict(self.X_val)
        plt.plot(
            self.dataproc.get_data_split().y_val.index,
            self.y_val,
            color="b",
            label="Actual",
        )
        plt.plot(
            self.dataproc.get_data_split().y_val.index,
            y_val_pred,
            color="r",
            label="Predicted",
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Values (Line Chart)")
        plt.legend()
        plt.show()

    def save_model(self, file_path: str = "model_svr.pkl"):
        logging.info("Saving model to %s", file_path)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        logging.info("Model saved to %s", file_path)

    def get_params(self):
        return self.model.get_params()
