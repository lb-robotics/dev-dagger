from abc import abstractmethod
from gpytorch.means import mean
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as tf
import gym

from src.model.policy_base import BasePolicy
from src.model.car_pole_novice import NoviceCartPole, test_novice
from src.experts.car_pole_expert import ExpertCartPole

MODEL_INPUT_FRAMES = 2  # only 2 frames are supported
SAVE_GIF = False


class DAgger():

    def __init__(self,
                 pi_expert: BasePolicy,
                 pi_novice: BasePolicy,
                 device='cuda') -> None:
        self.pi_expert = pi_expert
        self.pi_novice = pi_novice
        self.device = device
        self.train_obs = []
        self.train_actions = []
        self.expert_query_count = 0
        self.incrementally_add_data = True

    def rollout_expert(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.pi_expert.control(obs_t)

    def rollout_novice(self, obs_t: torch.Tensor) -> tuple:
        # both mean and variance are returned
        return self.pi_novice.control(obs_t)

    @abstractmethod
    def decision_rule(self):
        raise NotImplementedError

    def aggregate_data(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> None:
        self.train_actions.append(act_t.cpu().float().detach().squeeze())
        self.train_obs.append(obs_t.cpu())

    def update_novice(self, max_iters: int) -> float:
        # TODO: investigate the cuda memory use of this function
        train_x = torch.stack(self.train_obs).to(self.device)
        train_y = torch.stack(self.train_actions).to(self.device)

        if isinstance(self.pi_novice, NoviceCartPole):
            if self.incrementally_add_data:
                if self.pi_novice.controller is None:
                    self.pi_novice.create_model(train_x, train_y)
                else:
                    # Incrementally add data
                    # CNN params are fixed between two trainings
                    img_train = train_x.reshape((train_x.shape[0], -1))
                    self.pi_novice.controller.set_train_data(
                        img_train, train_y, strict=False)
            else:
                self.pi_novice.create_model(train_x, train_y)

        loss = self.pi_novice.train(max_iters, train_x=train_x, train_y=train_y)

        del train_x, train_y

        return loss

    def reset_expert(self) -> None:
        self.pi_expert.reset()


class VanillaDAgger(DAgger):

    def __init__(self,
                 pi_expert,
                 pi_novice,
                 epsilon=1.0,
                 discount=0.999) -> None:
        super().__init__(pi_expert, pi_novice)
        self.epsilon = epsilon
        self.discount_factor = discount

    def decision_rule(self, obs_t_novice: torch.Tensor,
                      obs_t_expert: torch.Tensor) -> torch.Tensor:
        action_novice = self.rollout_novice(
            obs_t_novice.unsqueeze(0).to(self.device))[0].squeeze()
        action_expert = self.rollout_expert(obs_t_expert)

        act_t = torch.empty_like(action_novice)

        choice = np.random.uniform(0, 1)
        if choice < self.epsilon:
            act_t = action_expert
        else:
            act_t = action_novice

        self.epsilon *= self.discount_factor
        self.expert_query_count += 1
        return act_t.cpu()


class DevDAgger(DAgger):

    def __init__(self,
                 pi_expert,
                 pi_novice,
                 uncertainty_thres,
                 initial_steps: int = 300) -> None:
        super().__init__(pi_expert, pi_novice)
        self.uncertainty_thres = uncertainty_thres
        self.initial_steps = initial_steps
        self.step_count = 0
        self.expert_query_count = initial_steps
        self.uncertainty_upper = None

    def decision_rule(self, obs_t_novice: torch.Tensor,
                      obs_t_expert: torch.Tensor) -> torch.Tensor:
        if self.step_count < self.initial_steps:
            action_expert = self.rollout_expert(obs_t_expert)
            self.step_count += 1
            return action_expert.cpu()

        self.step_count += 1

        action_novice, uncertainty_novice = self.rollout_novice(
            obs_t_novice.unsqueeze(0).to(self.device))
        action_novice = action_novice.squeeze()
        uncertainty_novice = uncertainty_novice.squeeze()

        if self.uncertainty_upper is None:
            self.uncertainty_upper = (
                self.pi_novice.controller.covar_module.outputscale +
                self.pi_novice.likelihood.noise).sqrt().item()

        print(uncertainty_novice.item(), self.uncertainty_upper)
        if uncertainty_novice > self.uncertainty_thres * self.uncertainty_upper:
            action_expert = self.rollout_expert(obs_t_expert)
            self.expert_query_count += 1
            return action_expert.cpu()
        else:
            return action_novice.cpu()


if __name__ == "__main__":
    import gym
    import os
    import imageio

    figures_path = os.path.join("img", "cart_pole_visual_vanilla")
    os.makedirs(figures_path, exist_ok=True)

    use_gt_states = False
    num_experiments = 5

    list_avg_test_duration = []
    list_expert_query_ratios = []
    list_expert_query_counts = []

    for experiment_id in range(num_experiments):
        print("----- Experiment {} -----".format(experiment_id))
        env = gym.make('CartPole-v1')
        novice = NoviceCartPole(num_frames=2, use_gt_states=use_gt_states)
        expert = ExpertCartPole()

        # dagger = VanillaDAgger(pi_expert=expert, pi_novice=novice, discount=0.99)
        dagger = DevDAgger(
            pi_expert=expert,
            pi_novice=novice,
            uncertainty_thres=0.4,
            initial_steps=300)

        num_episodes = 10
        timesteps_per_episode = 100
        total_samples = 0
        for episode in range(num_episodes):
            state = env.reset()
            dagger.reset_expert()

            iterator = tqdm(range(timesteps_per_episode))
            for t in iterator:
                total_samples += 1
                if not use_gt_states:
                    img = env.render(mode="rgb_array")
                    img_transform = tf.Compose(
                        [tf.ToTensor(), tf.Resize((64, 64))])
                    img = img_transform(img.astype(np.float32) / 255.0)
                if t == 0:
                    pid = dagger.pi_expert.control(torch.from_numpy(state))
                    if not use_gt_states:
                        stack_img = torch.cat([img, img], dim=0)
                        dagger.aggregate_data(stack_img, pid)
                    else:
                        dagger.aggregate_data(torch.from_numpy(state), pid)
                else:
                    if not use_gt_states:
                        if t >= MODEL_INPUT_FRAMES - 1:
                            stack_img = torch.cat([prev_img, img], dim=0)
                        pid = dagger.decision_rule(
                            obs_t_novice=stack_img,
                            obs_t_expert=torch.from_numpy(state))
                        action = 0 if pid < 0 else 1
                        dagger.aggregate_data(stack_img, pid)
                    else:
                        pid = dagger.decision_rule(
                            obs_t_novice=torch.from_numpy(state),
                            obs_t_expert=torch.from_numpy(state))
                        action = 0 if pid < 0 else 1
                        dagger.aggregate_data(torch.from_numpy(state), pid)

                if not use_gt_states:
                    prev_img = img.clone()

                if t % int(timesteps_per_episode / 3) == 0:
                    loss = dagger.update_novice(100)
                    iterator.set_postfix(loss=loss)

                action = 0 if pid < 0 else 1

                state, reward, done, info = env.step(action)
                if done:
                    break

        query_ratio = dagger.expert_query_count / total_samples
        print("Total expert query count:", dagger.expert_query_count)
        print("Total expert query ratio:", query_ratio)
        list_expert_query_ratios.append(query_ratio)
        list_expert_query_counts.append(dagger.expert_query_count)

        #######################################
        ### Evaluation and visualization
        #######################################
        mean_duration = 0.0
        num_trials = 10
        with torch.no_grad():
            for i in range(num_trials):
                state = env.reset()
                gif = []
                for t in range(100):
                    if SAVE_GIF or not use_gt_states:
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

                        pid, uncertainty = dagger.pi_novice.control(
                            stack_img.clone().unsqueeze(0))
                        prev_img = img.clone()
                    else:
                        pid, uncertainty = dagger.pi_novice.control(
                            torch.tensor(state.reshape((1, -1))).to('cuda'))

                    # print(action.cpu().item(), uncertainty.cpu().item())
                    action = 0 if pid < 0 else 1
                    choice = np.random.uniform(0, 1)
                    if choice < 0.2:
                        state, reward, done, info = env.step(
                            env.action_space.sample())
                    else:
                        state, reward, done, info = env.step(action)

                    if done:
                        break
                print("Episode finished after {} timesteps".format(t + 1))
                mean_duration += (t + 1)

                if SAVE_GIF:
                    record_path = os.path.join(figures_path,
                                               'rollout' + str(i) + '.gif')
                    imageio.mimwrite(record_path, gif)

        avg_test_duration = mean_duration / num_trials
        print("avg test duration:", avg_test_duration)
        list_avg_test_duration.append(avg_test_duration)

        env.close()

    print("Average test duration for {} iterations: {}".format(
        num_experiments, np.mean(list_avg_test_duration)))
    print("Average expert query ratio for {} iterations: {}".format(
        num_experiments, np.mean(list_expert_query_ratios)))
    print("Average expert query count for {} iterations: {}".format(
        num_experiments, np.mean(list_expert_query_counts)))
