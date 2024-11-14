# Comparison of Stock Price Prediction Models with and without Sentiment Analysis for IHSG Stocks

## Project Description
This project serves as the final thesis project for the writer. The primary goal is to compare the performance of stock price prediction models for IHSG (Indeks Harga Saham Gabungan) stocks using two different approaches:
1. Prediction based on **Return and Close** values only.
2. Prediction based on **Return and Close** values with **Sentiment Analysis**.

The project will involve experimenting with various machine learning models and techniques to understand if incorporating sentiment data improves the prediction accuracy of stock prices.

## Data Sources
- **Stock Data**: Stock data can be sourced from any reliable stock data provider, such as Yahoo Finance, Google Finance, or IDX (Indonesia Stock Exchange) providers.
- **Sentiment Data**: News articles can be manually scraped and labeled for sentiment or sourced from providers with pre-sentiment-tagged data.

## Structure
The project will contain the following components:
Dataset/
├── Evaluation/
├── News/
└── Stock/
Experiments/
Models/
├── Factory.py
├── ModelLR.py
├── ModelLSTM.py
├── ModelRidge.py
├── ModelSVR.py
├── ModelXGBoost.py
└── __init__.py
Process/
├── Dataproc.py
├── SentimentAnalysis.py
└── __init__.py
saved_models/
Utils/
└── WriteEvaluation.py
Web/

## Copyright
All rights reserved. This project is copyrighted by:
- **Zadosaadi Brahmantio Purwanto**
- **Muhammad Rizki Wiratama**
- **Widha Dwiyanti**

**Note**: When using any part of this project or open-source materials from it, please include proper attribution to the original authors as listed above.

## Dependencies
This project may use the following dependencies:
- Python (e.g., version 3.8 or higher)
- Libraries: Pandas, NumPy, Scikit-Learn, TensorFlow/PyTorch (for modeling), Selectolax (for web scraping), and any other relevant packages.

## Installation
1. Clone this repository.
2. Install the necessary libraries using:
   ```bash
   pip install -r requirements.txt
