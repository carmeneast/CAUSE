import numpy as np
from typing import List


class MetricTracker:
    """
    Tracks metrics from different batches or epochs and computes the final value
    """

    def __init__(self, metric='nll'):
        self.metric: str = metric
        self.values: List[float] = []
        self.counts: List[int] = []
        self.val: float = np.nan
        self.sum: float = 0
        self.count: int = 0
        self.avg: float = np.nan

    def reset(self):
        self.__init__()
        return self

    def update(self, val, n=1):
        self.values.append(val)
        self.counts.append(n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self
