import torch
import numpy as np

from robotics.CURL_SAC.utils import QNetwork, PolicyNetwork, Normalizer
from robotics.CURL_SAC.curl import CURL

import os
from copy import deepcopy


class SACAgent:
    def __init__(self,
                 goal_space_shape,
                 action_space_shape,
                 action_ranges,
                 hidden_dim,
                 gamma,
                 tau,
                 alpha,
                 q_lr,
                 alpha_lr,
                 policy_lr,
                 device='cuda',
                 mode='multi_env'):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = device

        # TODO: maybe add unique seeds for each environment acting
        self.seeds = [0, 1996]

        self.obs_norm = Normalizer(hidden_dim, multi_env=True if mode == 'multi_env' else False)
        self.goal_norm = Normalizer(goal_space_shape, multi_env=True if mode == 'multi_env' else False)

        self.q_network_1 = QNetwork(hidden_dim, goal_space_shape, action_space_shape).to(self.device)
        self.q_network_2 = QNetwork(hidden_dim, goal_space_shape, action_space_shape).to(self.device)
        self.q_1_opt = torch.optim.Adam(self.q_network_1.parameters(), lr=q_lr)
        self.q_2_opt = torch.optim.Adam(self.q_network_2.parameters(), lr=q_lr)

        self.target_q_network_1 = deepcopy(self.q_network_1)
        self.target_q_network_2 = deepcopy(self.q_network_2)

        self.policy_network = PolicyNetwork(hidden_dim, goal_space_shape,
                                            action_space_shape, action_ranges).to(self.device)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        self.curl = CURL(hidden_dim=hidden_dim, device=self.device)

        # alpha part
        self.alpha = alpha
        temp = torch.ones(1, requires_grad=True)
        self.log_alpha = temp.new_tensor([np.log(alpha)], dtype=torch.float, device=self.device, requires_grad=True)
        del temp

        self.target_entropy = -torch.prod(torch.Tensor((action_space_shape, )).to(self.device)).item()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, states, evaluate=False):
        observations = states['observation']
        if self.mode == 'single_env' and len(observations.shape) == 3:
            observations = observations[np.newaxis, :, :, :]
        goals = states['desired_goal'] - states['achieved_goal']

        if not evaluate:
            self.obs_norm.update_stats(observations)
            self.goal_norm.update_stats(goals)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations)).to(self.device)
        goals = torch.FloatTensor(self.goal_norm.normalize(goals)).to(self.device)

        observations = self.curl(observations)

        if evaluate:
            actions = self.policy_network.sample(observations, goals, evaluate=True)
        else:
            actions, _ = self.policy_network.sample(observations, goals)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        return actions.cpu().data.numpy()

    def train(self, batch, update_alpha=True):
        obs, goals, actions, rewards, next_obs, next_goals, dones = batch

        self.curl.train(obs)

        # normalize weights
        action_scale = self.policy_network.action_scale
        action_bias = self.policy_network.action_bias
        actions = (actions - action_bias) / action_scale
        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        goals = self.goal_norm.normalize(goals, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)
        next_goals = self.goal_norm.normalize(next_goals, device=self.device)

        obs = self.curl(obs)
        next_obs = self.curl(next_obs)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample(next_obs, next_goals)
            next_actions = (next_actions - action_bias) / action_scale
            target_q_1_next = self.target_q_network_1(next_obs, next_goals, next_actions)
            target_q_2_next = self.target_q_network_2(next_obs, next_goals, next_actions)
            min_q_target = torch.min(target_q_1_next, target_q_2_next)
            next_value_function = min_q_target - self.alpha * next_log_probs

            q_hat = rewards + (1 - dones) * self.gamma * next_value_function.detach()

        q_1_loss = torch.mean(0.5 * (self.q_network_1(obs, goals, actions) - q_hat) ** 2)
        q_2_loss = torch.mean(0.5 * (self.q_network_2(obs, goals, actions) - q_hat) ** 2)

        policy_actions, log_probs = self.policy_network.sample(obs, goals)
        policy_actions = (policy_actions - action_bias) / action_scale
        q_1 = self.q_network_1(obs, goals, policy_actions)
        q_2 = self.q_network_2(obs, goals, policy_actions)
        q = torch.min(q_1, q_2)
        # Try to minimize action components mean to reach the goal with min possible control
        policy_loss = torch.mean((self.alpha * log_probs) - q) + torch.mean((policy_actions / action_scale).pow(2))

        # backpropagation
        q_1_loss.backward()
        q_2_loss.backward()
        policy_loss.backward()

        # gradient step
        self.policy_opt.step()
        self.q_1_opt.step()
        self.q_2_opt.step()

        # zero gradients of optimizers
        self.policy_opt.zero_grad()
        self.q_1_opt.zero_grad()
        self.q_2_opt.zero_grad()

        if update_alpha:
            actions, log_probs = self.policy_network.sample(obs, goals)

            entropy_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            entropy_loss.backward()
            self.alpha_optim.step()
            self.alpha_optim.zero_grad()

            self.alpha = torch.clamp(self.log_alpha.exp(), min=1e-2, max=2)

        self._soft_update(self.target_q_network_1, self.q_network_1, self.tau)
        self._soft_update(self.target_q_network_2, self.q_network_2, self.tau)

        if update_alpha:
            return q_1_loss, q_2_loss, policy_loss, q.mean(), entropy_loss, self.alpha
        else:
            return q_1_loss, q_2_loss, policy_loss, q.mean()

    @staticmethod
    def _soft_update(target, online, tau):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_((1 - tau)*target_param.data + tau*param.data)

    def save_models(self, model_name):
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        if not os.path.exists(f'./weights/{model_name}'):
            os.mkdir(f'./weights/{model_name}')

        self.curl.save(f'./weights/{model_name}/curl_encoder.pt')

        self.obs_norm.save(f'./weights/{model_name}', 'obs')
        self.goal_norm.save(f'./weights/{model_name}', 'goal')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network_1.state_dict(), f'./weights/{model_name}/q_1_network.pt')
        torch.save(self.q_network_2.state_dict(), f'./weights/{model_name}/q_2_network.pt')
        torch.save(self.target_q_network_1.state_dict(), f'./weights/{model_name}/target_q_network_1.pt')
        torch.save(self.target_q_network_2.state_dict(), f'./weights/{model_name}/target_q_network_2.pt')

    def load_normalizer_parameters(self, model_name):
        self.obs_norm.load(f'./weights/{model_name}', 'obs')
        self.goal_norm.load(f'./weights/{model_name}', 'goal')

    def load_pretrained_models(self, model_name, evaluate=False):
        if not os.path.exists('./weights'):
            raise IOError("No ./weights folder to load pretrained models")
        if not os.path.exists(f'./weights/{model_name}'):
            raise IOError(f"No ./weights/{model_name} folder to load pretrained model")

        self.curl.load(f"./weights/{model_name}/curl_encoder.pt")
        self.load_normalizer_parameters(model_name)
        self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}/policy_network.pt",
                                                       map_location=lambda storage, loc: storage))

        if evaluate:
            return

        self.q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/q_1_network.pt",
                                                    map_location=lambda storage, loc: storage))
        self.q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/q_2_network.pt",
                                                    map_location=lambda storage, loc: storage))
        self.target_q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network_1.pt",
                                                           map_location=lambda storage, loc: storage))
        self.target_q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network_2.pt",
                                                           map_location=lambda storage, loc: storage))
