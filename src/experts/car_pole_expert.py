import numpy as np
import gym
import imageio
import os

import torch

from src.model.policy_base import BasePolicy


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ExpertCartPole(BasePolicy):

    def __init__(self, P: float = 0.1, I: float = 0.01, D: float = 0.5) -> None:
        # (x, x_dot, theta, theta_dot)
        self.desired_state = np.array([0, 0, 0, 0])
        # pick the target state attributes to control
        self.desired_mask = np.array([0, 0, 1, 0])

        self.P = P
        self.I = I
        self.D = D

        self.integral = 0
        self.prev_error = 0

    def control(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.numpy()
        error = observation - self.desired_state

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        pid = np.dot(
            self.P * error + self.I * self.integral + self.D * derivative,
            self.desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        return torch.tensor(pid)

    def reset(self) -> None:
        self.integral = 0
        self.prev_error = 0

    def train(self):
        pass


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    controller = ExpertCartPole()
    figures_path = os.path.join("img", "cart_pole_expert")
    os.makedirs(figures_path, exist_ok=True)
    epsilon = 0.2

    for i_episode in range(20):
        state = env.reset()
        gif = []
        for t in range(50):
            img = env.render(mode="rgb_array")
            gif.append(img)
            action = controller.control(torch.from_numpy(state))

            choice = np.random.uniform(0, 1)
            if choice < epsilon:
                state, reward, done, info = env.step(env.action_space.sample())
            else:
                state, reward, done, info = env.step(action.numpy())

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        record_path = os.path.join(
            figures_path,
            'random_' + str(epsilon) + '_rollout' + str(i_episode) + '.gif')
        imageio.mimwrite(record_path, gif)

    env.close()
