import logging
import os
from datetime import datetime
from Process import Dataproc, DatasetConfig
from Models import ModelFactory
from Utils import WriteEvaluation


# Define the directory for logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist

# Create a filename based on the current date and time
log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))

# Configure logging to output to the file with the specified format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Also output to console
    ],
)

if __name__ == "__main__":
    model_type = "XGBoost"  # Pastikan nama konsisten dengan di Factory
    config = DatasetConfig(file_path=r"Dataset\Stock\IHSG_Stock_Clean.csv")
    dataproc = Dataproc(config)

    # Cek model_type dan buat instance ModelXGBoost
    if model_type == "XGBoost":
        model = ModelXGBoost(dataproc)

        # Hyperparameter tuning dengan Optuna
        model.optimize_hyperparameters(n_trials=100)  # Sesuaikan jumlah trial
        model._set_params(
            model.params
        )  # Mengatur hyperparameter terbaik yang ditemukan ke model

    else:
        # Jika model lain, pakai ModelFactory
        model = ModelFactory.use_model(model_type, dataproc)

    # Latih model
    model.create_model(use_returns=True, return_type="relative")
    model.show_summary()

    # Plot predictions sebagai line chart
    model.plot_prediction()  # Fungsi ini sekarang sudah disesuaikan ke line chart

    # Menyimpan hasil evaluasi
    WriteEvaluation(model.evaluation, model_type)
