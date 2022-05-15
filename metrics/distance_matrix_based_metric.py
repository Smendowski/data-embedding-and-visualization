import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from .metric import Metric, measure_time


class DistanceMatrixBasedMetric(Metric):

    def __init__(self, df_data: pd.DataFrame, df_labels: pd.Series):
        self._df_data = df_data
        self._df_labels = df_labels
        self._labels = self._get_labels()

    def _get_labels(self) -> list:
        return list(set(self._df_labels.unique()))

    def _calculate_mean_distance(
            self, matrix_of_distances: np.ndarray) -> np.float32:
        numerator = matrix_of_distances.sum() / 2
        denominator = np.count_nonzero(matrix_of_distances)

        return numerator / denominator

    @measure_time
    def calculate_metric(self) -> np.float32:
        labels = self._get_labels()
        dist_inner_mean = dist_outter_mean = 0
        for label in labels:
            mask = self._df_labels.astype(np.int32).isin([label])
            current_class_points = self._df_data[mask].iloc[:, :-1]
            other_classes_points = self._df_data[~mask].iloc[:, :-1]

            current_class_points_values = current_class_points.values
            other_classes_points_values = other_classes_points.values

            dist_current_class = distance_matrix(current_class_points_values,
                                                 current_class_points_values)

            dist_other_classes = distance_matrix(current_class_points_values,
                                                 other_classes_points_values)

            dist_inner_mean += \
                self._calculate_mean_distance(dist_current_class)
            dist_outter_mean += \
                self._calculate_mean_distance(dist_other_classes)

        return dist_inner_mean / dist_outter_mean
