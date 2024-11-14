import pandas as pd


class SentimentAnalysis:
    def __init__(self, news: pd.DataFrame, flag: int) -> None:
        self.news = news
        self.flag = flag
        self.proccessed_data = self.scenario(flag)

    def scenario(self, flag: int):

        if flag == 1:
            df_avg_sentimen = (
                self.news.groupby("tanggal")["skor_sentimen"].mean().reset_index()
            )
            df_avg_sentimen_clean = df_avg_sentimen.dropna(subset=["skor_sentimen"])

            df_avg_sentimen_clean.set_index("tanggal", inplace=True)

        return df_avg_sentimen_clean

    def get_sentiment(self):
        pass
