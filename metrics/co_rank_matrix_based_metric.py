import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def _visualize_co_rank_matrix(self) -> None:
        plt.matshow(np.log(self._co_rank_matrix+1e-2))

    def _calculate_average_normalized_agreement(
            self, k: np.int) -> np.float32:
        """
        [QNX - Average Normalized Agreement Between K-ary Neighborhoods]

        QNX measure the quality of data embedding technique in terms of
        how well it preserves the local neighborhood around observations.
        For a given value of k, the k closest points for each sample are
        retrieved. QXN is the number of shared neighbors between original
        dimensionality and the reduced one, additionally normalized by k.
        Simply, QNX yields values in the range from 0 to 1. 1 means that
        the neighborhoods are supremely preserved. QNX value is close to 0
        when there is no neighborhood preservation.

        [Source]
        Lee, J. A., & Verleysen, M. (2009).
        Quality assessment of dimensionality reduction: Rank-based criteria.

        [Reference to the implementation in R]
        https://github.com/jlmelville/quadra/blob/master/R/neighbor.R
        """
        numerator = np.sum(self._co_rank_matrix[:k, :k])
        denominator = k * len(self._co_rank_matrix)

        return numerator / denominator

    def _calculate_rescaled_agreement(self, k: np.int) -> np.float32:
        """
        [RNX - Rescaled Agreement Between K-ary Neighborhoods]

        RXN is the scaled version of QNX. RNX measures the quality of
        data embedding technique in terms of the shared number of k-NN.
        RNX yields values in the range from 0 to 1. 1 means that the
        neighborhoods are supremely preserved. On the contrary, 0 means
        that the neighborhoods are not preserved and the embedding resemble
        the random one.

        [Source]
        Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M.
        (2013). Type 1 and 2 mixtures of Kullback-Leibler divergences as
        cost functions in dimensionality reduction based on similarity
        preservation.

        [Reference to the implementation in R]
        https://github.com/jlmelville/quadra/blob/master/R/neighbor.R
        """
        n = len(self._co_rank_matrix)
        numerator = \
            (self._calculate_average_normalized_agreement(k) * (n - 1)) - k
        denominator = n - 1 - k

        return numerator / denominator

    def _calculate_area_under_rnx_curve(self):
        """
        [AUC - Area Under RNX Curve]

        AUC of 1 indicates perfect neighborhood preservation.
        AUC of 0 indicates no neighborhood preservation due to random
        results.

        [Source]
        Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
        Multi-scale similarities in stochastic neighbour embedding:
        Reducing dimensionality while preserving both local and global
        structure.

        [Reference to the implementation in R]
        https://github.com/jlmelville/quadra/blob/master/R/neighbor.R
        """
        n = len(self._co_rank_matrix)
        numerator = 0
        denominator = 0
        qnx_crm_sum = 0
        for k in range(1, n-2):
            qnx_crm_sum += np.sum(
                np.sum(self._co_rank_matrix[(k-1), :k]) +
                np.sum(self._co_rank_matrix[:k, (k-1)]) -
                self._co_rank_matrix[(k-1), (k-1)]
            )

            qnx_crm = qnx_crm_sum / (k * len(self._co_rank_matrix))
            rnx_crm = ((qnx_crm * (n - 1)) - k) / (n - 1 - k)
            numerator += rnx_crm / k
            denominator += 1 / k

        return numerator / denominator

    def _calculate_qnx_and_rnx_values(self, k: int):
        qnx, rnx = [], []
        for k_val in range(k):
            qnx.append(self._calculate_average_normalized_agreement(k=k_val))
            rnx.append(self._calculate_rescaled_agreement(k=k_val))

        return qnx, rnx

    @measure_time
    def calculate(self) -> dict:
        self._co_rank_matrix = \
            self._compute_co_rank_matrix(self._df_data, self._df_embedding)

        self._qnx, self._rnx = self._calculate_qnx_and_rnx_values(
            min(10_000, self._df_data.shape[0])
        )

        self._area_under_rnx = self._calculate_area_under_rnx_curve()

        return json.dumps({
            'QNX': self._qnx,
            'RNX': self._rnx,
            'AREA_UNDER_RNX_CURVE': self._area_under_rnx
        })
