# app.py

import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
import logging
import os
from datetime import datetime
from Process.Dataproc import Dataproc, DatasetConfig
from Models.Factory import ModelFactory
from Utils.WriteEvaluation import WriteEvaluation

# Logging Setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# Streamlit App Interface
st.title("Dynamic Stock Model Prediction and Evaluation")
st.sidebar.header("Configuration")

# Sidebar Parameters
model_type = st.sidebar.selectbox(
    "Model Type", ["Ridge", "SVR", "Xgboost", "LSTM", "LR"]
)
use_returns = st.sidebar.checkbox("Use Returns?", value=False)
return_type = st.sidebar.selectbox("Return Type", ["absolute", "relative", "log"])
split_ratio = st.sidebar.slider("Split Ratio", 0.5, 0.9, 0.8)
n_in = st.sidebar.number_input(
    "Look-back Period (n_in)", min_value=1, max_value=50, value=10
)
sentiment_scenario = st.sidebar.selectbox("Sentiment Scenario", [1, 2, 3])
drop_columns = st.sidebar.checkbox("Drop Columns?", value=True)
drop_columns_list = st.sidebar.multiselect(
    "Columns to Drop",
    options=["Change%", "Volume", "Open", "Close"],
    default=["Change%", "Volume"],
)

# Date Inputs
start_date = st.sidebar.date_input("Start Date", datetime(2014, 1, 3))
end_date = st.sidebar.date_input("End Date", datetime(2024, 8, 6))

# Sentiment Analysis Option
use_sentiment = st.sidebar.checkbox("Include Sentiment Analysis?", value=False)
file_path_sentiment = None  # Default to None if sentiment is not used
if use_sentiment:
    file_path_sentiment = st.sidebar.text_input(
        "Sentiment Dataset Path", "Dataset/News/News_Kompas.csv"
    )

# Custom Parameters (JSON Input)
use_custom_params = st.sidebar.checkbox("Use Custom Parameters?", value=False)
custom_params_json = "{}"  # Default empty JSON
if use_custom_params:
    custom_params_json = st.sidebar.text_area(
        "Custom Parameters (JSON format)",
        value='{"C": 300, "epsilon": 0.01, "gamma": "auto", "kernel": "rbf", "degree": 3}',
    )

# Main Content
st.write("### Model Configuration")
st.write("Current model configuration is displayed below:")
st.write(
    {
        "Model Type": model_type,
        "Use Returns": use_returns,
        "Return Type": return_type,
        "Split Ratio": split_ratio,
        "Look-back Period": n_in,
        "Sentiment Scenario": sentiment_scenario,
        "Drop Columns": drop_columns,
        "Drop Columns List": drop_columns_list,
        "Start Date": start_date,
        "End Date": end_date,
        "Include Sentiment": use_sentiment,
        "Sentiment Dataset Path": file_path_sentiment if use_sentiment else "Not Used",
        "Use Custom Parameters": use_custom_params,
        "Custom Params": custom_params_json if use_custom_params else "Default",
    }
)

# Parse JSON for Custom Parameters
if use_custom_params:
    try:
        params = json.loads(custom_params_json)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for custom parameters. Please correct it.")
        params = None
else:
    params = None

if st.button("Run Model"):
    # Define configuration and initialize data processing
    config = DatasetConfig(
        file_path="Dataset/Stock/IHSG_Stock_Clean.csv",
        file_path_sentiment=file_path_sentiment if use_sentiment else None,
        split_ratio=split_ratio,
        n_in=n_in,
        sentiment_scenario=sentiment_scenario,
        drop_columns=drop_columns,
        drop_columns_list=drop_columns_list,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    # Process data
    dataproc = Dataproc(config)

    # Select and initialize model
    if use_custom_params and params:
        model = ModelFactory.use_model(model_type, dataproc, params)
    else:
        model = ModelFactory.use_model(model_type, dataproc)

    # Create model and generate predictions
    model.create_model(use_returns=use_returns, return_type=return_type)
    model.show_summary()
    fig = model.plot_prediction()

    # Display Model Summary and Predictions
    st.write("### Model Summary")
    st.write(model.evaluation)

    st.write("### Model Predictions")
    st.pyplot(fig)

    # Evaluation
    WriteEvaluation(
        model.evaluation, model_type, model.get_params(), use_returns, return_type
    )
    st.write("Model Evaluation written to logs.")

    # Option to save the model
    if st.button("Save Model"):
        model.save_model()
        st.success("Model saved successfully!")
