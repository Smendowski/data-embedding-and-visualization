from abc import ABC, abstractmethod
from time import perf_counter
from functools import wraps
import numpy as np


def measure_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__qualname__.split('.')[0]} calculation "
              f"took {elapsed:.2f} seconds.")
        return result
    return inner


class Metric(ABC):

    @abstractmethod
    def _get_labels(self) -> list:
        pass

    @abstractmethod
    def calculate_metric(self, df_data, df_labels) -> np.float32:
        pass
