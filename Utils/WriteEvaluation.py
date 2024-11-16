import pandas as pd
import os
import csv
import json
from datetime import datetime
import logging


class WriteEvaluation:
    def __init__(
        self,
        evaluation: pd.DataFrame,
        model_name: str,
        extra_params: dict = None,
        returns: bool = False,
        return_type: str = None,
    ) -> None:
        self.evaluation = evaluation
        self.model_name = model_name
        self.extra_params = extra_params or {}
        self.returns = returns
        self.return_type = return_type if returns else None
        self._write_to_csv()

    def _write_to_csv(self):
        if self.evaluation is None or self.evaluation.empty:
            raise ValueError("Evaluation DataFrame is None or empty.")

        folder_path = os.path.join("Dataset", "Evaluation", self.model_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "evaluation.csv")

        # Prepare evaluation data with extra_params as a JSON string
        evaluation_data = {
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **self.evaluation.iloc[0].to_dict(),
            "extra_params": json.dumps(
                self.extra_params
            ),  # Convert extra_params to JSON string
            "returns": self.returns,
            "return_type": self.return_type,
        }

        # Determine the field names based on evaluation data
        fieldnames = list(evaluation_data.keys())

        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                writer.writeheader()
            writer.writerow(evaluation_data)

        logging.info("Evaluation saved to %s", file_path)
