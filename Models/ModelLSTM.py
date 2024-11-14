from Process.Dataproc import Dataproc, DatasetConfig
from dataclasses import dataclass
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
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
from typing import Dict
import logging


@dataclass
class ModelEvaluation:
    rmse: float
    mse: float
    r2: float
    mae: float
    mape: float


class ModelLSTM:
    def __init__(self, dataproc: Dataproc):
        self.dataproc = dataproc
        self.return_type = ""
        self.X_train = self.dataproc.get_scaled_data().X_train_scaled_transposed
        self.X_val = self.dataproc.get_scaled_data().X_val_scaled_transposed
        self.y_train = None
        self.y_val = None
        self.model = self._set_params_and_model()
        self.history = None
        self.evaluation = None
        logging.info("ModelLSTM initialized")

    def _set_params_and_model(self) -> Sequential:
        model_lstm = Sequential()
        model_lstm.add(
            Bidirectional(
                LSTM(
                    128,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                )
            )
        )
        model_lstm.add(Dropout(0.1))
        model_lstm.add(LSTM(128, activation="relu", return_sequences=False))
        model_lstm.add(Dense(128))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer="adam", loss="mse", metrics=["mse"])
        return model_lstm

    def create_model(self, use_returns: bool = False, return_type: str = ""):
        self.return_type = return_type
        logging.info(
            "Creating model with use_returns=%s and return_type='%s'",
            use_returns,
            return_type,
        )
        data_split = self.dataproc.get_data_split()

        if use_returns:
            self._set_return_data(return_type)
        else:
            self.y_train = data_split.y_train
            self.y_val = data_split.y_val
            logging.info(
                "Data split without returns. Shapes - X_train: %s, y_train: %s",
                self.X_train.shape,
                self.y_train.shape,
            )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=20, verbose=1, restore_best_weights=True
        )
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=200,
            batch_size=32,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
        )

        self.history = history
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

    def save_model(self, file_path: str = "model_lstm.pkl"):
        logging.info("Saving model to %s", file_path)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        logging.info("Model saved to %s", file_path)


if __name__ == "__main__":
    config = DatasetConfig(file_path=r"Dataset\Stock\IHSG_Stock_Clean.csv")
    dataproc = Dataproc(config)
    # params = {"kernel": "rbf", "C": 1.0, "gamma": 0.1}
    model = ModelLSTM(dataproc)
    model.create_model(use_returns=False)
    model.show_summary()
    model.plot_prediction()

    print("Predicted values (first 4):", model.predict(model.X_val)[:4])
    print("Actual values (first 4):", model.y_val[:4])
