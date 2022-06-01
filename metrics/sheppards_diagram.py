import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from .metric import Metric, measure_time
import numpy as np
import matplotlib.pyplot as plt


class ShepardDiagram(Metric):
    def __init__(self, df_data: pd.DataFrame, df_embedding: pd.DataFrame, df_labels: pd.Series, n_sample: int = 30):
        self._df_data = df_data
        self._df_embedding = df_embedding
        self._df_labels = df_labels
        self.original_distances = None
        self.embedding_distances = None
        self._n_sample = n_sample

    def _get_labels(self) -> list:
        return list(set(self._df_labels.unique()))

    def _get_sample(self):
        indexes = []
        labels = self._get_labels()
        for label in labels:
            label_sample = self._df_labels[self._df_labels == label].sample(self._n_sample)
            indexes = indexes + list(label_sample.index.values.astype(int))

        return indexes

    @staticmethod
    def _delete_diagonal(matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

    @measure_time
    def calculate(self):
        indexes = self._get_sample()
        original_sample = self._df_data.iloc[indexes]
        embedded_sample = self._df_embedding.iloc[indexes]

        original_distances = euclidean_distances(original_sample)
        original_distances = self._delete_diagonal(original_distances)
        embedded_distances = euclidean_distances(embedded_sample)
        embedded_distances = self._delete_diagonal(embedded_distances)

        self.original_distances = np.reshape(original_distances,
                                             (original_distances.shape[0] * original_distances.shape[1], 1))
        self.embedding_distances = np.reshape(embedded_distances,
                                              (embedded_distances.shape[0] * embedded_distances.shape[1], 1))

    def show(self):
        plt.scatter(self.original_distances, self.embedding_distances, alpha=0.7)
        plt.xlabel('Input distance')
        plt.ylabel('Output distance')
        plt.grid()
        plt.show()
