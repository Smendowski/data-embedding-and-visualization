import numpy as np
import pandas as pd
from scipy.stats.mstats import spearmanr
from .metric import Metric, measure_time
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances


class SpearmanCorrelationBasedMetric(Metric):

    def __init__(self, df_data: pd.DataFrame, df_labels: pd.Series,
                 df_embedding: pd.DataFrame):
        self._df_data = df_data
        self._df_labels = df_labels
        self._df_embedding = df_embedding
        self._labels = self._get_labels()

    def _get_labels(self) -> list:
        return list(set(self._df_labels.unique()))

    @measure_time
    def calculate(self) -> np.float32:
        target_classes = self._labels
        target_classes_indices = dict.fromkeys(target_classes)

        coefficients = []

        for target_class in target_classes:
            target_classes_indices[target_class] = \
                self._df_labels[self._df_labels == target_class].index.tolist()

            df_target = \
                self._df_data.iloc[target_classes_indices[target_class], :]
            df_target_values = df_target.values

            df_target_emb = \
                self._df_embedding.iloc[target_classes_indices[target_class], :]
            df_target_emb_values = df_target_emb.values

            dist_original = np.square(
                euclidean_distances(
                    df_target_values, df_target_values
                )
            ).flatten()

            dist_emb = np.square(
                euclidean_distances(
                    df_target_emb_values, df_target_emb_values
                )
            ).flatten()

            coef_emb, p_emb = spearmanr(dist_original, dist_emb)
            coefficients.append(coef_emb)

        if np.nan in coefficients:
            coefficients = np.array(coefficients)
            return np.mean(
                coefficients[np.logical_not(np.isnan(coefficients))]
            )
        else:
            return np.mean(coefficients)
