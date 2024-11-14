
import pandas as pd
import os
import csv
from datetime import datetime
import logging

class WriteEvaluation:
    def __init__(self, evaluation: pd.DataFrame, model_name: str) -> None:
        self.evaluation = evaluation
        self.model_name = model_name
        self._write_to_csv()

    def _write_to_csv(self):
        if self.evaluation is None or self.evaluation.empty:
            raise ValueError("Evaluation DataFrame is None or empty.")
        
        folder_path = os.path.join("Dataset", "Evaluation", self.model_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "evaluation.csv")
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["date_time", "rmse", "mse", "r2", "mae", "mape"])
            if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                writer.writeheader()
            writer.writerow({"date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **self.evaluation.iloc[0].to_dict()})
        logging.info("Evaluation saved to %s", file_path)