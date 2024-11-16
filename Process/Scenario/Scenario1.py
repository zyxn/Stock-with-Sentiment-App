# Scenario1.py

import pandas as pd


class Scenario1:
    def __init__(self, news: pd.DataFrame) -> None:
        self.news = news
        self.result = self.process()

    def process(self) -> pd.DataFrame:
        df_avg_sentimen = (
            self.news.groupby("tanggal")["skor_sentimen"].mean().reset_index()
        )
        df_avg_sentimen_clean = df_avg_sentimen.dropna(subset=["skor_sentimen"])
        df_avg_sentimen_clean.set_index("tanggal", inplace=True)
        return df_avg_sentimen_clean
