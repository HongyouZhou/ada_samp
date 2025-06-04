from abc import ABC, abstractmethod

import torch as th
from torch import Tensor

from src.samplers.registry import register_noise_sampler


class NoiseSampler(ABC):
    @abstractmethod
    def sample(self, shape: th.Size) -> Tensor:
        pass

@register_noise_sampler("gaussian")
class GaussianNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.randn(shape)

@register_noise_sampler("uniform")
class UniformNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.rand(shape) * 2 - 1

@register_noise_sampler("rademacher")
class RademacherNoiseSampler(NoiseSampler):
    def sample(self, shape: th.Size) -> Tensor:
        return th.randint(0, 2, shape, dtype=th.float32) * 2 - 1
