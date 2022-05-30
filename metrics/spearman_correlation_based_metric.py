import numpy as np
import pandas as pd
from scipy.stats.mstats import spearmanr
from .metric import Metric, measure_time
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances


class EmbeddingNotSupportedError(Exception):
    """Custom error raised when provided embedding neame is not supported"""


class SpearmanCorrelationBasedMetric(Metric):

    def __init__(self, df_data: pd.DataFrame, df_labels: pd.Series,
                 emb_name: str):
        self._df_data = df_data
        self._df_labels = df_labels
        self._labels = self._get_labels()
        self._emb_name = self._validate_embedding_name(emb_name)

    def _validate_embedding_name(self, emb_name: str):
        if emb_name in ["t-SNE", "UMAP", "IVHD"]:
            return emb_name
        else:
            raise EmbeddingNotSupportedError(f"{emb_name} is not supported!")

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

            if self._emb_name == "t-SNE":
                tsne_emb = TSNE(
                    n_components=2, perplexity=42
                ).fit_transform(df_target_values)
            elif self._emb_name == "UMAP":
                raise NotImplementedError
            elif self._emb_name == "IVHD":
                raise NotImplementedError

            dist_original = np.square(
                euclidean_distances(
                    df_target_values, df_target_values
                )
            ).flatten()

            dist_tsne = np.square(
                euclidean_distances(
                    tsne_emb, tsne_emb
                )
            ).flatten()

            coef_tsne, p_tsne = spearmanr(dist_original, dist_tsne)
            coefficients.append(coef_tsne)

        return np.mean(coefficients)
