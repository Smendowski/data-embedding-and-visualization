import numpy as np
import pandas as pd
import numba
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked
from .metric import Metric, measure_time


class CoRankMatrixBasedMetric(Metric):
    def __init__(
            self,
            df_data: pd.DataFrame,
            df_embedding: pd.DataFrame,
            df_labels: pd.Series
    ):
        self._df_data = df_data
        self._df_embedding = df_embedding
        self._df_labels = df_labels
        self._labels = self._get_labels()
        self._n_jobs = -1 if df_data.shape[0] > 1000 else 1
        self._preference = "threads"

    def _get_labels(self) -> list:
        return list(set(self._df_labels.unique()))

    def _compute_distance_matrix(self, data: np.ndarray):
        distance_matrix = np.zeros(
            (len(data), len(data)), dtype='float32'
        )

        col = 0
        for chunk in pairwise_distances_chunked(data, n_jobs=self._n_jobs):
            distance_matrix[col:col+len(chunk)] = chunk
            col += len(chunk)

        return distance_matrix

    def _compute_rank_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        order = Parallel(self._n_jobs, prefer=self._preference)(
            delayed(np.argsort)(row) for row in distance_matrix
        )

        rank = Parallel(self._n_jobs, prefer=self._preference)(
            delayed(np.argsort)(row) for row in order
        )

        return np.vstack(rank)

    @staticmethod
    @numba.njit(fastmath=True)
    def _populate_co_rank_matrix_row(
        co_rank_matrix: np.ndarray,
        row_idx_to_populate: np.int16,
        num_of_columns_to_populate: np.int16,
        rank_matrix_hd: np.ndarray,
        rank_matrix_ld: np.ndarray
    ) -> np.ndarray:
        for col in range(num_of_columns_to_populate):
            k = rank_matrix_hd[row_idx_to_populate, col]
            l = rank_matrix_ld[row_idx_to_populate, col]
            co_rank_matrix[k, l] += 1

        return co_rank_matrix

    def _compute_co_rank_matrix(
            self, data_hd: np.ndarray, data_ld: np.ndarray) -> np.ndarray:
        high_dimensional_distances = self._compute_distance_matrix(data_hd)
        low_dimensional_distances = self._compute_distance_matrix(data_ld)

        rank_matrix_hd = self._compute_rank_matrix(high_dimensional_distances)
        rank_matrix_ld = self._compute_rank_matrix(low_dimensional_distances)

        co_rank_matrix = np.zeros(
            rank_matrix_hd.shape, dtype='int16'
        )

        for row in range(co_rank_matrix.shape[0]):
            co_rank_matrix = self._populate_co_rank_matrix_row(
                co_rank_matrix=co_rank_matrix,
                row_idx_to_populate=row,
                num_of_columns_to_populate=co_rank_matrix.shape[0],
                rank_matrix_hd=rank_matrix_hd,
                rank_matrix_ld=rank_matrix_ld
            )

        return co_rank_matrix[1:, 1:].T

    @measure_time
    def calculate(self):
        return self._compute_co_rank_matrix(self._df_data, self._df_embedding)
