import abc
import torch
from typing import Union


class BasePolicy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def control(self, observation: torch.Tensor) -> Union[torch.Tensor, tuple]:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, iters: int, optimizer: torch.optim.Optimizer,
              train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        raise NotImplementedError
