import torch
import numpy as np

from robotics.TD3.utils import QNetwork, PolicyNetwork, Normalizer

import os
from copy import deepcopy


class TD3Agent:
    def __init__(self,
                 observation_dim,
                 goal_dim,
                 action_dim,
                 action_ranges,
                 gamma,
                 tau,
                 q_lr,
                 policy_lr,
                 image_as_state=True,
                 device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = device
        include_conv = image_as_state
        self.action_low, self.action_high = action_ranges

        # TODO: maybe add unique seeds for each environment acting
        self.seeds = [0, 1996]

        self.obs_norm = Normalizer(observation_dim, multi_env=True if mode == 'multi_env' else False)
        self.goal_norm = Normalizer(goal_dim, multi_env=True if mode == 'multi_env' else False)

        self.q_network_1 = QNetwork(observation_dim, goal_dim, action_dim, include_conv=include_conv).to(self.device)
        self.target_q_network_1 = deepcopy(self.q_network_1)

        self.q_network_2 = QNetwork(observation_dim, goal_dim, action_dim, include_conv=include_conv).to(self.device)
        self.target_q_network_2 = deepcopy(self.q_network_2)

        self.policy_network = PolicyNetwork(observation_dim, goal_dim, action_dim, include_conv=include_conv).to(self.device)
        self.target_policy_network = deepcopy(self.policy_network)

        self.exploration_noise = torch.distributions.Normal(0, 0.1)
        self.smoothing_noise = torch.distributions.Normal(0, 0.2)

        self.q_1_opt = torch.optim.Adam(self.q_network_1.parameters(), lr=q_lr)
        self.q_2_opt = torch.optim.Adam(self.q_network_2.parameters(), lr=q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

    def select_action(self, states, evaluate=False):
        observations = states['observation']
        if self.mode == 'single_env' and len(observations.shape) == 3:
            observations = observations[np.newaxis, :, :, :]
        goals = states['desired_goal']

        if not evaluate:
            self.obs_norm.update_stats(observations)
            self.goal_norm.update_stats(goals)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations)).to(self.device)
        goals = torch.FloatTensor(self.goal_norm.normalize(goals)).to(self.device)

        actions = self.policy_network(observations, goals)
        if not evaluate:
            actions += self.exploration_noise.sample()
            actions = torch.clamp(actions, min=self.action_low, max=self.action_high)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        return actions.cpu().data.numpy()

    def train(self, batch, iteration, writer):
        obs, goals, actions, rewards, next_obs, next_goals, dones = batch

        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        goals = self.goal_norm.normalize(goals, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)
        next_goals = self.goal_norm.normalize(next_goals, device=self.device)

        # Q network update
        self.q_1_opt.zero_grad()
        self.q_2_opt.zero_grad()

        with torch.no_grad():
            next_actions = self.target_policy_network(next_obs, next_goals)
            smoothing_noise = self.smoothing_noise.sample()
            smoothing_noise = torch.clamp(smoothing_noise, min=-0.5, max=0.5)
            next_actions = torch.clamp(next_actions + smoothing_noise, min=self.action_low, max=self.action_high)

            target_q_1_next = self.target_q_network_1(next_obs, next_goals, next_actions)
            target_q_2_next = self.target_q_network_2(next_obs, next_goals, next_actions)
            target_q_next = torch.min(target_q_1_next, target_q_2_next)

            q_hat = rewards + (1 - dones) * self.gamma * target_q_next.detach()

        q_1_loss = torch.mean(0.5 * (self.q_network_1(obs, goals, actions) - q_hat) ** 2)
        q_2_loss = torch.mean(0.5 * (self.q_network_2(obs, goals, actions) - q_hat) ** 2)
        q_1_loss.backward()
        self.q_1_opt.step()

        q_2_loss.backward()
        self.q_2_opt.step()

        if iteration % 2 == 0:
            # Policy network update
            policy_actions = self.policy_network(obs, goals)
            q = self.q_network_1(obs, goals, policy_actions)

            policy_loss = -torch.mean(q) + torch.mean(policy_actions.pow(2))

            policy_loss.backward()
            self.policy_opt.step()
            self.policy_opt.zero_grad()

            self._soft_update(self.target_q_network_1, self.q_network_1, self.tau)
            self._soft_update(self.target_q_network_2, self.q_network_2, self.tau)

            self._soft_update(self.target_policy_network, self.policy_network, self.tau)

            writer.add_scalar("Policy_loss", policy_loss, iteration)

        writer.add_scalar("Q1_loss", q_1_loss, iteration)
        writer.add_scalar("Q2_loss", q_1_loss, iteration)

    @staticmethod
    def _soft_update(target, online, tau):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_((1 - tau)*target_param.data + tau*param.data)

    def save_models(self, model_name):
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        if not os.path.exists(f'./weights/{model_name}'):
            os.mkdir(f'./weights/{model_name}')

        self.obs_norm.save(f'./weights/{model_name}', 'obs')
        self.goal_norm.save(f'./weights/{model_name}', 'goal')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network_1.state_dict(), f'./weights/{model_name}/q_network_1.pt')
        torch.save(self.q_network_2.state_dict(), f'./weights/{model_name}/q_network_1.pt')
        torch.save(self.target_q_network_1.state_dict(), f'./weights/{model_name}/target_q_network_1.pt')
        torch.save(self.target_q_network_2.state_dict(), f'./weights/{model_name}/target_q_network_1.pt')

    def load_normalizer_parameters(self, model_name):
        self.obs_norm.load(f'./weights/{model_name}', 'obs')
        self.goal_norm.load(f'./weights/{model_name}', 'goal')

    def load_pretrained_models(self, model_name, evaluate=False):
        if not os.path.exists('./weights'):
            raise IOError("No ./weights folder to load pretrained models")
        if not os.path.exists(f'./weights/{model_name}'):
            raise IOError(f"No ./weights/{model_name} folder to load pretrained model")

        if evaluate:
            self.load_normalizer_parameters(model_name)
            self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}/policy_network.pt",
                                                map_location=lambda storage, loc: storage))
            return

        self.q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/q_network_1.pt",
                                                    map_location=lambda storage, loc: storage))
        self.q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/q_network_2.pt",
                                                    map_location=lambda storage, loc: storage))
        self.target_q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network_1.pt",
                                                           map_location=lambda storage, loc: storage))
        self.target_q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network_2.pt",
                                                           map_location=lambda storage, loc: storage))
