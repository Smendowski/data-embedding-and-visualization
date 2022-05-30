import pandas as pd
from sklearn.manifold import trustworthiness
from .metric import Metric, measure_time


class TrustworthinessBasedMetric(Metric):
    def __init__(self, df_data: pd.DataFrame, df_embedding: pd.DataFrame):
        self._df_data = df_data
        self._df_embedding = df_embedding
        self._n_metrics = ['euclidean', 'cosine']
        self._n_range = [5, 10, 15, 30, 50, 100, 150, 300, 500]

    def _get_labels(self):
        ...

    @measure_time
    def calculate(self):
        return {
            metric: {
                str(n): round(trustworthiness(
                    self._df_data, self._df_embedding,
                    n_neighbors=n, metric=metric
                ), 3) for n in self._n_range
            } for metric in self._n_metrics
        }
