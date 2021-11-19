from gpytorch.means import mean
import numpy as np
import torch
import gpytorch
import gym
import torchvision.transforms as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio
import os

from src.model.visual_controller import GPController, VisualGPController
from src.experts.car_pole_expert import ExpertCartPole

VISUALIZE_EXPERT_DATA = False
MODEL_INPUT_FRAMES = 2  # only 2 frames are supported


class NoviceCartPole():

    def __init__(self, num_frames=1, use_gt_states=False) -> None:
        self.controller = None
        self.use_gt_states = use_gt_states
        self.num_frames = num_frames

    def create_model(
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
            img_train = img_train.reshape((img_train.shape[0], -1))
            self.controller = VisualGPController(
                train_x=img_train,
                train_y=action_train,
                likelihood=self.likelihood,
                num_frames=self.num_frames)

    def control(self, img):
        # reshape img to (# imgs, -1)
        img = img.reshape((img.shape[0], -1))
        out_distribution = self.controller(img)
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
            train_x = train_x.reshape((train_x.shape[0], -1))
            output = self.controller(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()


if __name__ == "__main__":
    figures_path = os.path.join("img", "cart_pole_visual_novice")
    os.makedirs(figures_path, exist_ok=True)

    env = gym.make('CartPole-v1')
    use_gt_states = False
    novice = NoviceCartPole(
        num_frames=MODEL_INPUT_FRAMES, use_gt_states=use_gt_states)
    """ Collect demonstration dataset """
    train_inputs = []
    train_actions = []
    epsilon = 0.2
    for i_episode in range(30):
        expert = ExpertCartPole()
        state = env.reset()
        for t in range(100):
            if not use_gt_states:
                img = env.render(mode="rgb_array")
                img_transform = tf.Compose([tf.ToTensor(), tf.Resize((64, 64))])
                img = img_transform(img.astype(np.float32) / 255.0)
                if t >= MODEL_INPUT_FRAMES - 1:
                    stack_img = torch.cat([prev_img, img], dim=0)
                    train_inputs.append(stack_img)
                prev_img = img.clone()
            else:
                train_inputs.append(torch.tensor(state))

            action = expert.control(state)
            if t >= MODEL_INPUT_FRAMES - 1:
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

    # Use the adam optimizer
    optimizer = torch.optim.Adam(novice.controller.parameters(), lr=0.1)

    novice.train(
        100,
        optimizer,
        train_x=train_inputs.to('cuda'),
        train_y=train_actions.to('cuda'))
    """ Evaluation and visualization """
    novice.controller.eval()
    novice.likelihood.eval()
    mean_duration = 0.0
    num_trials = 10
    with torch.no_grad():
        for i in range(num_trials):
            state = env.reset()
            gif = []
            for t in range(100):
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
