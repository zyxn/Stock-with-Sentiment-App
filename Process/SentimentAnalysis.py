# SentimentAnalysis.py

import pandas as pd
from .Scenario.Scenario1 import Scenario1  # Import Scenario1 class
from .Scenario.Scenario2 import Scenario2  # Import Scenario1 class


class SentimentAnalysis:
    def __init__(self, news: pd.DataFrame, flag: int) -> None:
        self.news = news
        self.flag = flag
        self.processed_data = self.scenario(flag)

    def scenario(self, flag: int) -> pd.DataFrame:
        if flag == 1:
            return Scenario1(self.news).result
        if flag == 2:
            return Scenario2(self.news).result
        # Add additional scenarios here as needed

    def get_sentiment(self):
        pass
