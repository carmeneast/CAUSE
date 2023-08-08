import math
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BasisFunction(metaclass=ABCMeta):
    def __init__(self, p):
        super().__init__()
        self._p = p

    def log_prob(self, value):
        return self._p.log_prob(value)

    def cdf(self, value):
        return self._p.cdf(value)

    @property
    @abstractmethod
    def maximum(self):
        pass


class Unity(nn.Module):
    """
    Basis function that returns:
    { exp(1) at x
    { 0 otherwise
    """
    def __init__(self):
        super().__init__(self)
        self.maximum = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    @staticmethod
    def log_prob(x):
        if isinstance(x, float):
            return torch.zeros(1)
        else:
            return torch.zeros_like(x)

    @staticmethod
    def cdf(x):
        if isinstance(x, float):
            return torch.tensor([x])
        else:
            return x


class Normal(nn.Module, BasisFunction):
    def __init__(self, mu, sigma):
        nn.Module.__init__(self)

        self.mu = nn.Parameter(torch.as_tensor(mu).float(), requires_grad=False)
        self.sigma = nn.Parameter(torch.as_tensor(sigma).float(), requires_grad=False)
        BasisFunction.__init__(self, torch.distributions.Normal(self.loc, self.scale))

    @property
    def maximum(self):
        return 1 / (2 * math.pi) ** 0.5 / self.sigma


class Uniform(nn.Module, BasisFunction):
    def __init__(self, low, high):
        nn.Module.__init__(self)

        self.low = nn.Parameter(torch.as_tensor(low).float(), requires_grad=False)
        self.high = nn.Parameter(torch.as_tensor(high).float(), requires_grad=False)
        BasisFunction.__init__(self, torch.distributions.Uniform(self.low, self.high))

    @property
    def maximum(self):
        return 1 / (self.high - self.low)
