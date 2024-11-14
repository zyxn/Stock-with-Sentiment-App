import logging
import os
from datetime import datetime
from Process.Dataproc import Dataproc, DatasetConfig
from Models.Factory import ModelFactory
from Utils.WriteEvaluation import WriteEvaluation

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
    model_type = "Xgboost"
    config = DatasetConfig(
        file_path=r"Dataset\Stock\IHSG_Stock_Clean.csv",
        # file_path_sentiment=r"Dataset\News\News_Kompas.csv", #Uncomment if you want to use sentiment
        split_ratio=0.8,
        n_in=10,
        drop_columns=True,
        drop_columns_list=["Change%", "Volume"],  # Kolom yang akan di-drop
        start_date="2014-01-03",
        end_date="2024-08-06",
    )
    dataproc = Dataproc(config)

    # params = {"kernel": "linear", "C": 1.0} #custom params
    model = ModelFactory.use_model(model_type, dataproc)  # , params untuk parameter model
    model.create_model(use_returns=True, return_type="log") # return_type bisa "absolute" atau "relative" atau "log"
    model.show_summary()
    model.plot_prediction()
    WriteEvaluation(model.evaluation, model_type)


# TODO
# load model -- Done
# Show in Streamlit -- TBA
# Add Sentiment Dataset -- Done
# seperate Sentiment Model -- TBA
# Combine Sentiment and Stock Data -- DONE
# add function filter time data -- DONE
# More Advance Write Evaluation for etl to csv and show in streamlit -- TBA
