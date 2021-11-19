from abc import abstractmethod
import numpy as np
import torch

from src.model.policy_base import BasePolicy
from src.model.car_pole_novice import NoviceCartPole
from src.experts.car_pole_expert import ExpertCartPole


class DAgger():

    def __init__(self, pi_expert: BasePolicy, pi_novice: BasePolicy) -> None:
        self.pi_expert = pi_expert
        self.pi_novice = pi_novice
        self.train_obs = []
        self.train_actions = []

    def rollout_expert(self, obs_t):
        return self.pi_expert.control(obs_t)

    def rollout_novice(self, obs_t):
        # both mean and variance are returned
        return self.pi_novice.control(obs_t)

    @abstractmethod
    def decision_rule(self):
        raise NotImplementedError


class VanillaDAgger(DAgger):

    def __init__(self, pi_expert, pi_novice, epsilon=0.2) -> None:
        super().__init__(pi_expert, pi_novice)
        self.epsilon = epsilon

    def decision_rule(self, obs_t):
        choice = np.random.uniform(0, 1)
        if choice < self.epsilon:
            pass


class DevDAgger(DAgger):

    def __init__(self, pi_expert, pi_novice) -> None:
        super().__init__(pi_expert, pi_novice)


if __name__ == "__main__":
    novice = NoviceCartPole(num_frames=2, use_gt_states=True)
    expert = ExpertCartPole()
    dev_dagger = DevDAgger(pi_expert=expert, pi_novice=novice)