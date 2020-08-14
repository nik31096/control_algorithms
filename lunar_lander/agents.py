import torch
import numpy as np

from lunar_lander.utils import QNetwork, PolicyNetwork, Normalizer

import os
from copy import deepcopy


class DDPGAgent:
    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 q_lr,
                 policy_lr,
                 device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = device

        self.obs_norm = Normalizer(observation_space_shape, multi_env=True if mode == 'multi_env' else False)

        self.q_network = QNetwork(observation_space_shape, action_space_shape).to(self.device)
        self.target_q_network = deepcopy(self.q_network)

        self.policy_network = PolicyNetwork(observation_space_shape, action_space_shape, action_ranges).to(self.device)
        self.target_policy_network = deepcopy(self.policy_network)

        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

    def select_action(self, states, noise=False, evaluate=False):
        observations = states
        if self.mode == 'single_env' and len(observations.shape) == 3:
            observations = observations[np.newaxis, :, :, :]

        if not evaluate:
            self.obs_norm.update_stats(observations)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations)).to(self.device)

        actions = self.policy_network.sample(observations, add_noise=noise, evaluate=evaluate)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        if 'cuda' in self.device:
            return actions.cpu().data.numpy()
        else:
            return actions.data.numpy()

    def train(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        # normalize weights before Q network
        action_scale = self.policy_network.action_scale
        action_bias = self.policy_network.action_bias
        actions = (actions - action_bias) / action_scale
        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)

        # Q network update
        self.q_opt.zero_grad()

        with torch.no_grad():
            next_actions = self.target_policy_network.sample(next_obs, add_noise=False, evaluate=False)
            next_actions = (next_actions - action_bias) / action_scale
            target_q_next = self.target_q_network(next_obs, next_actions)

        q_hat = rewards + (1 - dones) * self.gamma * target_q_next.detach()
        q = self.q_network(obs, actions)

        q_loss = torch.mean(0.5 * (q - q_hat) ** 2)
        q_loss.backward()
        self.q_opt.step()

        # Policy network update
        policy_actions = self.policy_network.sample(obs, add_noise=False, evaluate=False)
        norm_policy_actions = (policy_actions - action_bias) / action_scale
        q = self.q_network(obs, norm_policy_actions)

        # Try to minimize action components mean to reach the goal with min possible control
        policy_loss = -torch.mean(q) + torch.mean(policy_actions.pow(2))

        policy_loss.backward()
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        self._soft_update(self.target_q_network, self.q_network, self.tau)
        self._soft_update(self.target_policy_network, self.policy_network, self.tau)

        return q_loss, policy_loss, q.mean()

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

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network.state_dict(), f'./weights/{model_name}/q_network.pt')
        torch.save(self.target_q_network.state_dict(), f'./weights/{model_name}/target_q_network.pt')
        torch.save(self.target_policy_network.state_dict(), f'./weights/{model_name}/target_policy_network.pt')

    def load_normalizer_parameters(self, model_name):
        self.obs_norm.load(f'./weights/{model_name}', 'obs')

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

        self.q_network.load_state_dict(torch.load(f"./weights/{model_name}/q_network.pt",
                                                  map_location=lambda storage, loc: storage))
        self.target_q_network.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network.pt",
                                                         map_location=lambda storage, loc: storage))
        self.target_policy_network.load_state_dict(torch.load(f"./weights/{model_name}/target_policy_network.pt",
                                                              map_location=lambda storage, loc: storage))


class SACAgent:
    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 alpha,
                 q_lr,
                 alpha_lr,
                 policy_lr,
                 device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = device

        # TODO: maybe add unique seeds for each environment acting
        self.seeds = [0, 1996]

        self.obs_norm = Normalizer(observation_space_shape, multi_env=True if mode == 'multi_env' else False)

        self.q_network_1 = QNetwork(observation_space_shape, action_space_shape).to(self.device)
        self.q_network_2 = QNetwork(observation_space_shape, action_space_shape).to(self.device)
        self.q_1_opt = torch.optim.Adam(self.q_network_1.parameters(), lr=q_lr)
        self.q_2_opt = torch.optim.Adam(self.q_network_2.parameters(), lr=q_lr)

        self.target_q_network_1 = deepcopy(self.q_network_1)
        self.target_q_network_2 = deepcopy(self.q_network_2)

        self.policy_network = PolicyNetwork(observation_space_shape, action_space_shape, action_ranges).to(self.device)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # alpha part
        self.alpha = alpha
        temp = torch.ones(1, requires_grad=True)
        self.log_alpha = temp.new_tensor([np.log(alpha)], dtype=torch.float, device=self.device, requires_grad=True)
        del temp

        self.target_entropy = -torch.prod(torch.Tensor((action_space_shape, )).to(self.device)).item()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, states, evaluate=False):
        observations = states['observation']

        if not evaluate:
            self.obs_norm.update_stats(observations)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations)).to(self.device)

        if evaluate:
            actions = self.policy_network.sample(observations, evaluate=True)
        else:
            actions, _ = self.policy_network.sample(observations)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        return actions.cpu().data.numpy()

    def train(self, batch, update_alpha=True):
        obs, actions, rewards, next_obs, dones = batch
        # normalize actions
        action_scale = self.policy_network.action_scale
        action_bias = self.policy_network.action_bias
        actions = (actions - action_bias) / action_scale
        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample(next_obs)
            next_actions = (next_actions - action_bias) / action_scale
            target_q_1_next = self.target_q_network_1(next_obs, next_actions)
            target_q_2_next = self.target_q_network_2(next_obs, next_actions)
            min_q_target = torch.min(target_q_1_next, target_q_2_next)
            next_value_function = min_q_target - self.alpha * next_log_probs

            q_hat = rewards + (1 - dones) * self.gamma * next_value_function.detach()

        q_1_loss = torch.mean(0.5 * (self.q_network_1(obs, actions) - q_hat) ** 2)
        q_2_loss = torch.mean(0.5 * (self.q_network_2(obs, actions) - q_hat) ** 2)

        policy_actions, log_probs = self.policy_network.sample(obs)
        policy_actions = (policy_actions - action_bias) / action_scale
        q_1 = self.q_network_1(obs, policy_actions)
        q_2 = self.q_network_2(obs, policy_actions)
        q = torch.min(q_1, q_2)
        # Try to minimize action components mean to reach the goal with min possible control
        policy_loss = torch.mean((self.alpha * log_probs) - q) + torch.mean((policy_actions / action_scale).pow(2))

        q_1_loss.backward()
        self.q_1_opt.step()

        q_2_loss.backward()
        self.q_2_opt.step()

        policy_loss.backward()
        self.policy_opt.step()

        # zero gradients of optimizers
        self.policy_opt.zero_grad()
        self.q_1_opt.zero_grad()
        self.q_2_opt.zero_grad()

        if update_alpha:
            actions, log_probs = self.policy_network.sample(obs)

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

        self.obs_norm.save(f'./weights/{model_name}', 'obs')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network_1.state_dict(), f'./weights/{model_name}/q_1_network.pt')
        torch.save(self.q_network_2.state_dict(), f'./weights/{model_name}/q_2_network.pt')
        torch.save(self.target_q_network_1.state_dict(), f'./weights/{model_name}/target_q_network_1.pt')
        torch.save(self.target_q_network_2.state_dict(), f'./weights/{model_name}/target_q_network_2.pt')

    def load_normalizer_parameters(self, model_name):
        self.obs_norm.load(f'./weights/{model_name}', 'obs')

    def load_pretrained_models(self, model_name, evaluate=False):
        if not os.path.exists('./weights'):
            raise IOError("No ./weights folder to load pretrained models")
        if not os.path.exists(f'./weights/{model_name}'):
            raise IOError(f"No ./weights/{model_name} folder to load pretrained model")

        self.load_normalizer_parameters(model_name)
        self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}/policy_network.pt"))

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


class NAFAgent:
    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 q_lr,
                 policy_lr,
                 device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = device

        self.obs_norm = Normalizer(observation_space_shape, multi_env=True if mode == 'multi_env' else False)

        self.q_network = QNetwork(observation_space_shape, action_space_shape).to(self.device)
        self.target_q_network = deepcopy(self.q_network)

        self.policy_network = PolicyNetwork(observation_space_shape, action_space_shape, action_ranges).to(self.device)
        self.target_policy_network = deepcopy(self.policy_network)

        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

    def select_action(self, states, noise=False, evaluate=False):
        observations = states
        if self.mode == 'single_env' and len(observations.shape) == 3:
            observations = observations[np.newaxis, :, :, :]

        if not evaluate:
            self.obs_norm.update_stats(observations)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations)).to(self.device)

        actions = self.policy_network.sample(observations, add_noise=noise, evaluate=evaluate)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        if 'cuda' in self.device:
            return actions.cpu().data.numpy()
        else:
            return actions.data.numpy()

    def train(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        # normalize weights before Q network
        action_scale = self.policy_network.action_scale
        action_bias = self.policy_network.action_bias
        actions = (actions - action_bias) / action_scale
        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)

        # Q network update
        self.q_opt.zero_grad()

        with torch.no_grad():
            next_actions = self.target_policy_network.sample(next_obs, add_noise=False, evaluate=False)
            next_actions = (next_actions - action_bias) / action_scale
            target_q_next = self.target_q_network(next_obs, next_actions)

        q_hat = rewards + (1 - dones) * self.gamma * target_q_next.detach()

        q_loss = torch.mean(0.5 * (self.q_network(obs, actions) - q_hat) ** 2)
        q_loss.backward()
        self.q_opt.step()

        # Policy network update
        policy_actions = self.policy_network.sample(obs, add_noise=False, evaluate=False)
        norm_policy_actions = (policy_actions - action_bias) / action_scale
        q = self.q_network(obs, norm_policy_actions)

        # Try to minimize action components mean to reach the goal with min possible control
        policy_loss = -torch.mean(q) + torch.mean(policy_actions.pow(2))

        policy_loss.backward()
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        self._soft_update(self.target_q_network, self.q_network, self.tau)
        self._soft_update(self.target_policy_network, self.policy_network, self.tau)

        return q_loss, policy_loss

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

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network.state_dict(), f'./weights/{model_name}/q_network.pt')
        torch.save(self.target_q_network.state_dict(), f'./weights/{model_name}/target_q_network.pt')
        torch.save(self.target_policy_network.state_dict(), f'./weights/{model_name}/target_policy_network.pt')

    def load_normalizer_parameters(self, model_name):
        self.obs_norm.load(f'./weights/{model_name}', 'obs')

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

        self.q_network.load_state_dict(torch.load(f"./weights/{model_name}/q_network.pt",
                                                  map_location=lambda storage, loc: storage))
        self.target_q_network.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network.pt",
                                                         map_location=lambda storage, loc: storage))
        self.target_policy_network.load_state_dict(torch.load(f"./weights/{model_name}/target_policy_network.pt",
                                                              map_location=lambda storage, loc: storage))
