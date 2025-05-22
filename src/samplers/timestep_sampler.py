from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch as th
    from torch import Tensor


class TimestepSampler(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> "Tensor":
        pass


class UniformTimestepSampler(TimestepSampler):
    def __init__(self, T: int):
        self.T = T

    def sample(self, batch_size: int) -> "Tensor":
        return th.randint(low=0, high=self.T, size=(batch_size,), device='cpu')


class LinearTimestepSampler(TimestepSampler):
    def __init__(self, T: int):
        self.T = T
        self.probs = th.linspace(1, self.T, self.T)
        self.probs = self.probs / self.probs.sum()

    def sample(self, batch_size: int) -> "Tensor":
        return th.multinomial(self.probs, batch_size, replacement=True)


class ImportanceTimestepSampler(TimestepSampler):
    def __init__(self, T: int, loss_history: "Tensor" = None):
        self.T = T
        self.loss_history = loss_history if loss_history is not None else th.ones(T)

    def update(self, new_losses: "Tensor"):
        self.loss_history = 0.9 * self.loss_history + 0.1 * new_losses

    def sample(self, batch_size: int) -> "Tensor":
        probs = self.loss_history / self.loss_history.sum()
        return th.multinomial(probs, batch_size, replacement=True)
