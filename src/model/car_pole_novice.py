import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
import gym
import torchvision.transforms as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio
import os

from src.model.visual_controller import (GPController, VisualGPController,
                                         VisVarGPCtrler)
from src.experts.car_pole_expert import ExpertCartPole
from src.model.policy_base import BasePolicy

VISUALIZE_EXPERT_DATA = False
MODEL_INPUT_FRAMES = 2  # only 2 frames are supported
UNCERTAINTY_ANALYSIS = True


class NoviceCartPole(BasePolicy):

    def __init__(self,
                 num_frames=1,
                 use_gt_states=False,
                 use_variational_GP=False,
                 device='cuda') -> None:
        self.controller = None
        self.use_gt_states = use_gt_states
        self.num_frames = num_frames
        self.use_variational_GP = use_variational_GP
        self.device = device

    def create_model(
        self,
        img_train: torch.Tensor,
        action_train: torch.Tensor,
    ) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            self.device)
        if self.use_gt_states:
            self.controller = GPController(
                train_x=img_train,
                train_y=action_train,
                likelihood=self.likelihood)
        else:
            img_train = img_train.reshape((img_train.shape[0], -1))
            if self.use_variational_GP:
                num_inducing = 256  # Can lower this if you want it to be faster
                if img_train.shape[0] < num_inducing:
                    print(
                        "Warning: input training points < inducing points required!"
                    )
                inducing_points = img_train[:num_inducing, :]
                self.controller = VisVarGPCtrler(
                    inducing_points=inducing_points, num_frames=self.num_frames)
            else:
                self.controller = VisualGPController(
                    train_x=img_train,
                    train_y=action_train,
                    likelihood=self.likelihood,
                    num_frames=self.num_frames)

    def control(self, observation: torch.Tensor) -> tuple:
        self.controller.eval()
        self.likelihood.eval()
        with torch.no_grad():
            # reshape observation to (# of observation, -1)
            observation = observation.reshape((observation.shape[0], -1))
            out_distribution = self.controller(observation)
        return out_distribution.mean, out_distribution.variance

    def train(self, iters, train_x, train_y) -> float:
        # Find optimal model hyperparameters
        self.controller.train()
        self.likelihood.train()

        if self.use_variational_GP:
            train_dataset = TensorDataset(
                train_x.reshape(train_x.size(0), -1), train_y)
            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True)

            # TODO: why does Adam cause gradient explosion
            optimizer = torch.optim.SGD([{
                'params': self.controller.parameters()
            }, {
                'params': self.likelihood.parameters()
            }],
                                        lr=0.01)

            mll = gpytorch.mlls.PredictiveLogLikelihood(
                self.likelihood, self.controller, num_data=train_y.size(0))

            iterator = range(iters)
            for i in iterator:
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                    optimizer.zero_grad()
                    output = self.controller(x_batch)
                    loss = -mll(output, y_batch)
                    print(loss.detach().cpu().item())
                    loss.backward()
                    optimizer.step()
        else:
            """ Using Exact GP """
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.controller)
            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.controller.parameters(), lr=0.1)

            iterator = range(iters)
            for i in iterator:
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                train_x = train_x.reshape((train_x.shape[0], -1))
                output = self.controller(train_x)
                # Calc loss and backprop derivatives
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
        return loss.detach().cpu().item()


if __name__ == "__main__":
    figures_path = os.path.join("img", "cart_pole_visual_novice")
    os.makedirs(figures_path, exist_ok=True)

    env = gym.make('CartPole-v1')
    use_gt_states = False
    novice = NoviceCartPole(
        num_frames=MODEL_INPUT_FRAMES,
        use_gt_states=use_gt_states,
        use_variational_GP=False)
    """ Collect demonstration dataset """
    train_inputs = []
    train_actions = []
    epsilon = 0.2
    for i_episode in range(10):
        expert = ExpertCartPole()
        state = env.reset()
        for t in range(100):
            action = expert.control(torch.from_numpy(state)).numpy()
            if not use_gt_states:
                img = env.render(mode="rgb_array")
                img_transform = tf.Compose([tf.ToTensor(), tf.Resize((64, 64))])
                img = img_transform(img.astype(np.float32) / 255.0)
                if t >= MODEL_INPUT_FRAMES - 1:
                    stack_img = torch.cat([prev_img, img], dim=0)
                    train_inputs.append(stack_img)
                    train_actions.append(
                        torch.tensor(action.astype(np.float32)))
                prev_img = img.clone()
            else:
                train_inputs.append(torch.tensor(state))
                train_actions.append(torch.tensor(action.astype(np.float32)))

            choice = np.random.uniform(0, 1)
            if choice < epsilon:
                state, reward, done, info = env.step(env.action_space.sample())
            else:
                state, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

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
    """ Train an imitation model based on the demonstration dataset """
    novice.create_model(train_inputs, train_actions)

    novice.train(
        100, train_x=train_inputs.to('cuda'), train_y=train_actions.to('cuda'))
    """ Evaluation and visualization """
    mean_duration = 0.0
    num_trials = 10

    all_novice_actions = []
    all_novice_uncertainties = []
    all_expert_actions = []
    with torch.no_grad():
        for i in range(num_trials):
            expert = ExpertCartPole()
            state = env.reset()
            gif = []
            for t in range(100):
                expert_action = expert.control(torch.from_numpy(state)).numpy()
                all_expert_actions.append(expert_action.item())

                img = env.render(mode="rgb_array")
                gif.append(img)
                if not use_gt_states:
                    img_transform = tf.Compose(
                        [tf.ToTensor(), tf.Resize((64, 64))])
                    img = img_transform(img.astype(np.float32) /
                                        255.0).to('cuda')
                    if t < MODEL_INPUT_FRAMES - 1:
                        stack_img = torch.cat([img, img], dim=0)
                    else:
                        stack_img = torch.cat([prev_img, img], dim=0)

                    action, uncertainty = novice.control(
                        stack_img.clone().unsqueeze(0))
                    # print(action, uncertainty)
                    all_novice_actions.append(action.detach().cpu().item())
                    all_novice_uncertainties.append(
                        uncertainty.detach().cpu().item())

                    prev_img = img.clone()
                else:
                    action, uncertainty = novice.control(
                        torch.tensor(state.reshape((1, -1))).to('cuda'))

                # print(action.cpu().item(), uncertainty.cpu().item())
                action = 0 if action < 0.5 else 1
                state, reward, done, info = env.step(action)

                if done:
                    break
            print("Episode finished after {} timesteps".format(t + 1))
            mean_duration += (t + 1)
            record_path = os.path.join(figures_path,
                                       'rollout' + str(i) + '.gif')
            imageio.mimwrite(record_path, gif)

    print("avg test duration:", mean_duration / num_trials)
    env.close()

    all_novice_actions = np.array(all_novice_actions)
    all_novice_uncertainties = np.array(all_novice_uncertainties)
    all_expert_actions = np.array(all_expert_actions)
    if UNCERTAINTY_ANALYSIS:
        plt.figure()
        plt.scatter(all_novice_actions, all_novice_uncertainties)
        plt.xlabel("novice action")
        plt.ylabel("novice uncertainty")
        plt.show()

        novice_binary_actions = (all_novice_actions > 0.5).astype(np.int32)
        correct_actions = novice_binary_actions == all_expert_actions.astype(
            np.int32)
        plt.figure()
        plt.scatter(all_novice_uncertainties, correct_actions)
        plt.xlabel("novice uncertainty")
        plt.ylabel("is novice action correct")
        plt.show()

        plt.figure()
        plt.scatter(all_novice_uncertainties,
                    np.abs(all_novice_actions - all_expert_actions))
        plt.xlabel("novice uncertainty")
        plt.ylabel("action error")
        plt.show()
