from enum import Enum
import numpy as np

class MetricType(Enum):
    Euclidean = 1
    Hamming = 2
    Manhattan = 3

class Metric:
    def __init__(self, metric_type: MetricType):
        self.metric_type = metric_type

    def __call__(self, x, y):
        if self.metric_type == MetricType.Euclidean:
            return np.linalg.norm(x - y)
        elif self.metric_type == MetricType.Hamming:
            return np.sum(x != y)
        elif self.metric_type == MetricType.Manhattan:
            return np.sum(np.abs(x - y))

