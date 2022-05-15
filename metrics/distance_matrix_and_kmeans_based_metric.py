import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from .metric import Metric, measure_time


class DistanceMatrixAndKMeansBasedMetric(Metric):

    def __init__(self, df_data: pd.DataFrame, df_labels: pd.Series):
        self._df_data = df_data
        self._df_labels = df_labels
        self._labels = self._get_labels()

    def _get_labels(self) -> list:
        return list(set(self._df_labels.unique()))

    def _get_points_from_each_class(self) -> list:
        points = []
        for label in self._labels:
            mask = (label == self._df_labels.astype(int))
            points.append(self._df_data[mask].values)

        return points

    def _get_mean_intra_class_distances(self, points) -> np.ndarray:
        mean_intra_class_distances = []
        for idx, _ in enumerate(points):
            d_matrix = distance_matrix(points[idx], points[idx])
            mean_distance = d_matrix.sum()/np.count_nonzero(d_matrix)
            mean_intra_class_distances.append(mean_distance)

        return mean_intra_class_distances

    def _get_mean_inter_class_distances(
            self, df_data_values: np.ndarray) -> np.ndarray:
        kmeans = KMeans(
            n_clusters=len(self._labels),
            random_state=0
        ).fit(df_data_values)
        cluster_centers = kmeans.cluster_centers_

        return distance_matrix(cluster_centers, cluster_centers)

    def remove_diagonal_distance_matrix(
            self, inter_class_distances: np.ndarray) -> np.ndarray:
        return inter_class_distances[
            ~np.eye(
                inter_class_distances.shape[0],
                dtype=bool
            )
        ].reshape(inter_class_distances.shape[0], -1)

    @measure_time
    def calculate_metric(self) -> np.float32:
        points = self._get_points_from_each_class()
        intra_class_distances = \
            self._get_mean_intra_class_distances(points)
        inter_class_distances = \
            self._get_mean_inter_class_distances(self._df_data.values)
        inter_class_distances = \
            self.remove_diagonal_distance_matrix(inter_class_distances)

        intra_to_inter_relations = []
        for i in range(inter_class_distances.shape[0]):

            intra_to_inter_relations.append(
                intra_class_distances[i] / inter_class_distances[i].mean()
            )

        return np.array(intra_to_inter_relations).mean()
