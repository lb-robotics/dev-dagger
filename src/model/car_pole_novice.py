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

VISUALIZE_EXPERT_DATA = True
VISUALIZE_NOVICE_UNCERTAINTIES = True
MODEL_INPUT_FRAMES = 2  # only 2 frames are supported
UNCERTAINTY_ANALYSIS = True

theta_scaling_factor = 100
theta_dot_scaling_factor = 10


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
        # self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(
        #     self.device)
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
            out_pred = self.likelihood(
                out_distribution)  # this is a ber distribution now
            confidence_lower, confidence_upper = out_pred.confidence_region()
        # return out_distribution.mean, (confidence_upper -
        #                                confidence_lower) / 4.0
        return out_pred.mean, out_pred.stddev  # should be equivalent to above
        # return out_pred.probs  # the prob of this point being 1

    def train(self, iters, train_x, train_y, use_tqdm: bool) -> float:
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

            iterator = range(iters) if not use_tqdm else tqdm(range(iters))
            for i in iterator:
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                    optimizer.zero_grad()
                    output = self.controller(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    if use_tqdm:
                        iterator.set_description(loss.detach().cpu().item())
        else:
            """ Using Exact GP """
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.controller)
            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.controller.parameters(), lr=0.1)

            iterator = range(iters) if not use_tqdm else tqdm(range(iters))
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
                if use_tqdm:
                    iterator.set_description(loss.detach().cpu().item())
        return loss.detach().cpu().item()


def data_collection(env: gym.Env,
                    num_episodes: int = 10,
                    epsilon: float = 0.2,
                    use_gt_states: bool = False):
    """ Collect demonstration dataset """
    train_inputs = []
    train_actions = []
    train_states = []
    for i_episode in range(num_episodes):
        expert = ExpertCartPole()
        state = env.reset()
        for t in range(100):
            pid = expert.control(torch.from_numpy(state))
            pid = pid.numpy()
            action = 0 if pid < 0 else 1
            if not use_gt_states:
                img = env.render(mode="rgb_array")
                img_transform = tf.Compose([tf.ToTensor(), tf.Resize((64, 64))])
                img = img_transform(img.astype(np.float32) / 255.0)
                if t >= MODEL_INPUT_FRAMES - 1:
                    stack_img = torch.cat([prev_img, img], dim=0)
                    train_inputs.append(stack_img)
                    train_actions.append(torch.tensor(pid.astype(np.float32)))
                    train_states.append(torch.tensor(state))
                prev_img = img.clone()
            else:
                train_inputs.append(torch.tensor(state))
                # train_actions.append(torch.tensor(action.astype(np.float32)))
                train_actions.append(torch.tensor(pid.astype(np.float32)))
                train_states.append(torch.tensor(state))

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
    train_states = torch.stack(train_states)
    return train_inputs, train_actions, train_states


def test_novice(env: gym.Env,
                novice: BasePolicy,
                figures_path: str,
                num_trials: int = 10,
                epsilon: float = 0,
                use_gt_states: bool = False,
                save_gif: bool = False):
    """ Evaluation and visualization """
    img_transform = tf.Compose([tf.ToTensor(), tf.Resize((64, 64))])

    mean_duration = 0.0
    all_novice_actions = []
    all_novice_uncertainties = []
    all_expert_actions = []
    all_states = []
    with torch.no_grad():
        for i in range(num_trials):
            expert = ExpertCartPole()
            state = env.reset()
            gif = []
            for t in range(100):
                expert_pid = expert.control(torch.from_numpy(state))
                expert_pid = expert_pid.numpy()

                all_expert_actions.append(expert_pid.item())

                if save_gif or not use_gt_states:
                    img = env.render(mode="rgb_array")
                    gif.append(img)

                if not use_gt_states:
                    img = img_transform(img.astype(np.float32) /
                                        255.0).to('cuda')
                    if t < MODEL_INPUT_FRAMES - 1:
                        stack_img = torch.cat([img, img], dim=0)
                    else:
                        stack_img = torch.cat([prev_img, img], dim=0)

                    action, uncertainty = novice.control(
                        stack_img.clone().unsqueeze(0))
                    # print(action, uncertainty)

                    prev_img = img.clone()
                else:
                    scaled_state = torch.tensor(state.reshape(
                        (1, -1))).to('cuda')
                    scaled_state[:, 2] *= theta_scaling_factor
                    scaled_state[:, 3] *= theta_dot_scaling_factor
                    action, uncertainty = novice.control(scaled_state[:, 2:])

                all_novice_actions.append(action.detach().cpu().item())
                all_novice_uncertainties.append(
                    uncertainty.detach().cpu().item())
                all_states.append(state)

                # print(action.cpu().item(), uncertainty.cpu().item())
                action = 0 if action < 0 else 1
                choice = np.random.uniform(0, 1)
                if choice < epsilon:
                    state, reward, done, info = env.step(
                        env.action_space.sample())
                else:
                    state, reward, done, info = env.step(action)

                if done:
                    break
            print("Episode finished after {} timesteps".format(t + 1))
            mean_duration += (t + 1)

            if save_gif:
                record_path = os.path.join(figures_path,
                                           'rollout' + str(i) + '.gif')
                imageio.mimwrite(record_path, gif)

    avg_test_duration = mean_duration / num_trials
    print("avg test duration:", avg_test_duration)

    return avg_test_duration, all_novice_actions, all_novice_uncertainties, all_expert_actions, all_states


def grid_uncertainty_visual(novice: BasePolicy,):
    all_novice_actions = []
    all_novice_uncertainties = []
    all_states = []
    with torch.no_grad():
        for theta in np.arange(-1, 1, 0.1):
            for theta_dot in np.arange(-2, 2, 0.2):
                state = np.array([theta, theta_dot])

                scaled_state = torch.tensor(
                    state.astype(np.float32).reshape((1, -1))).to('cuda')
                scaled_state[:, 0] *= theta_scaling_factor
                scaled_state[:, 1] *= theta_dot_scaling_factor
                action, uncertainty = novice.control(scaled_state)

                all_novice_actions.append(action.detach().cpu().item())
                all_novice_uncertainties.append(
                    uncertainty.detach().cpu().item())
                all_states.append(state)

    return all_novice_actions, all_novice_uncertainties, all_states


if __name__ == "__main__":
    figures_path = os.path.join("img", "cart_pole_visual_novice")
    os.makedirs(figures_path, exist_ok=True)

    num_experiments = 5

    list_avg_test_durations = []

    list_train_inputs = []
    list_train_actions = []

    for experiment_id in range(num_experiments):
        print("----- Experiment {} -----".format(experiment_id))

        env = gym.make('CartPole-v1')
        use_gt_states = False
        novice = NoviceCartPole(
            num_frames=MODEL_INPUT_FRAMES,
            use_gt_states=use_gt_states,
            use_variational_GP=False)

        train_inputs, train_actions, train_states = data_collection(
            env, num_episodes=10, epsilon=0, use_gt_states=use_gt_states)

        if VISUALIZE_EXPERT_DATA:
            fig = plt.figure()
            ax = plt.axes()
            sc = ax.scatter(
                train_states[:, 2], train_states[:, 3], c=train_actions)
            plt.colorbar(sc)
            plt.title("Training Expert Actions vs. State")
            plt.xlabel("theta")
            plt.ylabel("theta_dot")
            plt.savefig("img/expert_data_{}.png".format(experiment_id), dpi=300)
            # plt.show()

        #######################################
        ### Train an imitation model based on the demonstration dataset
        #######################################
        if use_gt_states:
            train_inputs[:, 2] *= theta_scaling_factor
            train_inputs[:, 3] *= theta_dot_scaling_factor
            novice.create_model(train_inputs[:, 2:], train_actions)
            novice.train(
                1000,
                train_x=train_inputs[:, 2:].to('cuda'),
                train_y=train_actions.to('cuda'))
        else:
            novice.create_model(train_inputs, train_actions)
            novice.train(
                1000,
                train_x=train_inputs.to('cuda'),
                train_y=train_actions.to('cuda'))

        #######################################
        ### Evaluation and visualization
        #######################################
        avg_test_duration, all_novice_actions, all_novice_uncertainties, all_expert_actions, all_states = test_novice(
            env,
            novice,
            figures_path,
            num_trials=10,
            epsilon=0.2,
            use_gt_states=use_gt_states)
        list_avg_test_durations.append(avg_test_duration)

        all_novice_actions = np.array(all_novice_actions)
        all_novice_uncertainties = np.array(all_novice_uncertainties)
        all_expert_actions = np.array(all_expert_actions)
        all_states = np.stack(all_states)

        env.close()

        if UNCERTAINTY_ANALYSIS:
            if VISUALIZE_NOVICE_UNCERTAINTIES:
                fig = plt.figure()
                ax = plt.axes()
                sc = ax.scatter(
                    all_states[:, 2],
                    all_states[:, 3],
                    c=all_novice_uncertainties)
                c_bar = plt.colorbar(sc)
                plt.title("Novice Uncertainty vs. State")
                plt.xlabel("theta")
                plt.ylabel("theta_dot")
                plt.savefig(
                    "img/novice_uncertainty_{}.png".format(experiment_id),
                    dpi=300)
                # plt.show()

                novice_actions_binary = (all_novice_actions >= 0.5)
                action_error = np.abs(novice_actions_binary -
                                      all_expert_actions)
                fig = plt.figure()
                ax = plt.axes()
                sc = ax.scatter(
                    all_states[:, 2], all_states[:, 3], c=action_error)
                c_bar = plt.colorbar(sc)
                plt.title("Novice Action Error vs. State")
                plt.xlabel("theta")
                plt.ylabel("theta_dot")
                plt.savefig(
                    "img/novice_error_{}.png".format(experiment_id), dpi=300)
                # plt.show()

            plt.figure()
            plt.scatter(all_novice_actions, all_novice_uncertainties)
            plt.xlabel("novice action")
            plt.ylabel("novice uncertainty")
            plt.savefig(
                "img/novice_action_uncertainty_{}.png".format(experiment_id),
                dpi=300)
            # plt.show()

            plt.figure()
            plt.scatter(all_novice_uncertainties,
                        np.abs(all_novice_actions - all_expert_actions))
            plt.xlabel("novice uncertainty")
            plt.ylabel("action error")
            plt.savefig(
                "img/error_vs_uncertainty_{}.png".format(experiment_id),
                dpi=300)
            # plt.show()

            # grid_novice_actions, grid_novice_uncertainties, grid_states = grid_uncertainty_visual(
            #     novice)

            # grid_novice_actions = np.array(grid_novice_actions)
            # grid_novice_uncertainties = np.array(grid_novice_uncertainties)
            # grid_states = np.stack(grid_states)
            # fig = plt.figure()
            # ax = plt.axes()
            # sc = ax.scatter(
            #     grid_states[:, 0],
            #     grid_states[:, 1],
            #     c=grid_novice_uncertainties)
            # c_bar = plt.colorbar(sc)
            # plt.title(
            #     "Grid Novice Uncertainty vs. State\n upper bound: {}".format(
            #         (novice.controller.covar_module.outputscale +
            #          novice.likelihood.noise).sqrt().item()))
            # plt.xlabel("theta")
            # plt.ylabel("theta_dot")
            # plt.savefig(
            #     "img/grid_novice_uncertainty_{}.png".format(experiment_id),
            #     dpi=300)

    print("Average test duration for {} iterations: {}".format(
        num_experiments, np.mean(list_avg_test_durations)))
