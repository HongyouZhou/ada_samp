from abc import ABC, abstractmethod

import torch as th
from torch import Tensor


class NoiseSampler(ABC):
    @abstractmethod
    def sample(self, shape: th.Size) -> Tensor:
        pass


class GaussianNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.randn(shape)


class UniformNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.rand(shape) * 2 - 1


class RademacherNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.randint(0, 2, shape, dtype=th.float32) * 2 - 1
