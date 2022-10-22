import random

__all__ = ["UniformSampler"]


class UniformSampler:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __float__(self):
        return self.sample()

    def sample(self):
        return random.uniform(self.low, self.high)
