import numpy as np
import gym


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ExpertCartPole():

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

    def control(self, state: np.ndarray) -> np.ndarray:
        error = state - self.desired_state

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        pid = np.dot(
            self.P * error + self.I * self.integral + self.D * derivative,
            self.desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        return action


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    controller = ExpertCartPole()

    for i_episode in range(20):
        state = env.reset()
        for t in range(500):
            env.render()
            action = controller.control(state)

            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
