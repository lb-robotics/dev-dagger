import numpy as np
import torch
import gpytorch
import gym
import torchvision.transforms as tf
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.model.visual_controller import GPController, VisualGPController
from src.experts.car_pole_expert import ExpertCartPole

VISUALIZE_EXPERT_DATA = False


class NoviceCartPole():

    def __init__(self, use_gt_states=False) -> None:
        self.controller = None
        self.use_gt_states = use_gt_states

    def fit_model(
        self,
        img_train: torch.Tensor,
        action_train: torch.Tensor,
    ) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.use_gt_states:
            self.controller = GPController(
                train_x=img_train,
                train_y=action_train,
                likelihood=self.likelihood)
        else:
            self.controller = VisualGPController(
                train_x=img_train,
                train_y=action_train,
                likelihood=self.likelihood)

    def control(self, img):
        out_distribution = self.controller(img)
        # y_pred = self.likelihood(out_distribution)
        return out_distribution.mean, out_distribution.variance

    def train(self, iters, optimizer, train_x, train_y) -> None:
        # Find optimal model hyperparameters
        self.controller.train()
        self.likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.controller)

        iterator = tqdm(range(iters))
        for i in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self.controller(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    use_gt_states = True
    novice = NoviceCartPole(use_gt_states=use_gt_states)

    train_inputs = []
    train_actions = []
    for i_episode in range(1):
        expert = ExpertCartPole()
        state = env.reset()
        for t in range(100):
            if not use_gt_states:
                img = env.render(mode="rgb_array")
                img_transform = tf.Compose([tf.ToTensor(), tf.Resize((64, 64))])
                img = img_transform(img.astype(np.float32) / 255.0)
                train_inputs.append(img)
            else:
                train_inputs.append(torch.tensor(state))

            action = expert.control(state)
            train_actions.append(torch.tensor(action.astype(np.float32)))

            state, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

    train_inputs = torch.stack(train_inputs)
    train_actions = torch.stack(train_actions)

    if VISUALIZE_EXPERT_DATA:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            train_inputs[:, 2],
            train_inputs[:, 3],
            train_actions,
            c=train_actions + 0.25)
        plt.show()

    novice.fit_model(train_inputs, train_actions)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(novice.controller.parameters(), lr=0.1)

    losses = novice.train(
        100,
        optimizer,
        train_x=train_inputs.to('cuda'),
        train_y=train_actions.to('cuda'))

    novice.controller.eval()
    novice.likelihood.eval()
    with torch.no_grad():
        for i in range(10):
            id = np.random.randint(0, train_inputs.shape[0])
            img = train_inputs[id:id + 1].to('cuda')
            out_action, out_uncertainty = novice.control(img)
            expert_action = train_actions[id]
            print("=================")
            print(out_action.cpu().item(), out_uncertainty.cpu().item())
            print("expert action:", expert_action.item())
