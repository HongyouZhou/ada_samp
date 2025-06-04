from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch as th

from src.samplers.registry import register_timestep_sampler

if TYPE_CHECKING:
    from torch import Tensor


class TimestepSampler(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> "Tensor":
        pass


@register_timestep_sampler("uniform")
class UniformTimestepSampler(TimestepSampler):
    def __init__(self, T: int):
        self.T = T

    def sample(self, batch_size: int) -> "Tensor":
        return th.randint(low=0, high=self.T, size=(batch_size,), device="cpu")

@register_timestep_sampler("linear")
class LinearTimestepSampler(TimestepSampler):
    def __init__(self, T: int, **kwargs):
        self.T = T
        self.probs = th.linspace(1, self.T, self.T)
        self.probs = self.probs / self.probs.sum()

    def sample(self, batch_size: int) -> "Tensor":
        return th.multinomial(self.probs, batch_size, replacement=True)

@register_timestep_sampler("importance")
class ImportanceTimestepSampler(TimestepSampler):
    def __init__(self, T: int, loss_history: "Tensor" = None, **kwargs):
        self.T = T
        self.loss_history = loss_history if loss_history is not None else th.ones(T)

    def update(self, new_losses: "Tensor"):
        self.loss_history = 0.9 * self.loss_history + 0.1 * new_losses

    def sample(self, batch_size: int) -> "Tensor":
        probs = self.loss_history / self.loss_history.sum()
        return th.multinomial(probs, batch_size, replacement=True)

@register_timestep_sampler("continuous")
class ContinuousTimeStepSampler(TimestepSampler):
    def __init__(self, num_timesteps, shift=1.0, logit_normal_enable=False, logit_normal_mean=0.0, logit_normal_std=1.0, **kwargs):
        self.num_timesteps = num_timesteps
        self.shift = shift
        self.logit_normal_enable = logit_normal_enable
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample(self, batch_size):
        if self.logit_normal_enable:
            t_linear = th.sigmoid(self.logit_normal_mean + self.logit_normal_std * th.randn((batch_size,), dtype=th.float))
        else:
            t_linear = 1 - th.rand((batch_size,), dtype=th.float)
        t = self.shift * t_linear / (1 + (self.shift - 1) * t_linear)
        return t * self.num_timesteps

    def __call__(self, batch_size):
        return self.sample(batch_size)
