import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..Process import Dataproc, DatasetConfig


# Fixture untuk konfigurasi dan objek Dataproc
@pytest.fixture
def sample_config():
    # Simpan dataset dummy

    # Buat konfigurasi
    return DatasetConfig(
        file_path=r"Dataset\Stock\IHSG_Stock_Clean.csv",
        file_path_sentiment=r"Dataset\News\Daily_Sentiment_Score.csv",
        split_ratio=0.8,
        n_in=2,
        sentiment_scenario=1,
        drop_columns=True,
        drop_columns_list=["Volume", "Change%"],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )


@pytest.fixture
def dataproc_instance(sample_config):
    return Dataproc(sample_config)


# Tes inisialisasi
def test_initialization(dataproc_instance):
    assert dataproc_instance.DATA.shape == (3, 4)  # Setelah kolom drop
    assert dataproc_instance.DATA_SUPERVISED is not None
    assert dataproc_instance.DATA_SENTIMENT is not None


# Tes pembacaan dataset
def test_read_dataset(dataproc_instance):
    data = dataproc_instance.DATA
    assert "Open" in data.columns
    assert data.index[0] == pd.Timestamp("2024-01-01")


def test_read_dataset_sentiment(dataproc_instance):
    data_sentiment = dataproc_instance.DATA_SENTIMENT
    assert data_sentiment.shape == (3, 1)
    assert "Sentiment_Score" in data_sentiment.columns


# Tes pembagian data
def test_split_data(dataproc_instance):
    split_data = dataproc_instance.data_split
    assert split_data.X_train.shape[0] == 2  # 80% dari 3 data
    assert split_data.X_val.shape[0] == 1


# Tes normalisasi data
def test_scale_data(dataproc_instance):
    scaled_data = dataproc_instance.scaled_data
    assert scaled_data.X_train_scaled.shape == (2, 8)  # 2 timestep, 4 fitur
    assert scaled_data.X_val_scaled.shape == (1, 8)


# Tes perhitungan returns
def test_calculate_returns(dataproc_instance):
    returns = dataproc_instance.returns
    assert "y_train" in returns
    assert len(returns["y_train"].absolute_return) == 2  # 2 data dalam train set
