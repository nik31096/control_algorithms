import torch
import numpy as np

from robotics.DDPG.utils import QNetwork, PolicyNetwork, Normalizer

import os


class DDPGAgent:
    def __init__(self,
                 observation_space_shape,
                 goal_space_shape,
                 action_space_shape,
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

        # TODO: maybe add unique seeds for each environment acting
        self.seeds = [0, 1996]

        self.obs_norm = Normalizer(observation_space_shape, multi_env=True if mode == 'multi_env' else False)
        self.goal_norm = Normalizer(goal_space_shape, multi_env=True if mode == 'multi_env' else False)

        self.q_network = QNetwork(observation_space_shape, goal_space_shape,
                                  action_space_shape, include_conv=include_conv).to(self.device)
        self.target_q_network = QNetwork(observation_space_shape, goal_space_shape,
                                         action_space_shape, include_conv=include_conv).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.policy_network = PolicyNetwork(observation_space_shape, goal_space_shape,
                                            action_space_shape, action_ranges, include_conv=include_conv).to(self.device)
        self.target_policy_network = PolicyNetwork(observation_space_shape,
                                                   goal_space_shape,
                                                   action_space_shape,
                                                   action_ranges,
                                                   include_conv=include_conv).to(self.device)
        self.online_policy_network = PolicyNetwork(observation_space_shape,
                                                   goal_space_shape,
                                                   action_space_shape,
                                                   action_ranges,
                                                   include_conv=include_conv)
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())

        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

    def select_action(self, states, noise=False, evaluate=False):
        observations = states['observation']
        if self.mode == 'single_env' and len(observations.shape) == 3:
            observations = observations[np.newaxis, :, :, :]
        desired_goals = states['desired_goal']
        achieved_goals = states['achieved_goal']
        goals = desired_goals  # - achieved_goals

        if not evaluate:
            self.obs_norm.update_stats(observations)
            self.goal_norm.update_stats(goals)

        observations = torch.FloatTensor(self.obs_norm.normalize(observations))
        goals = torch.FloatTensor(self.goal_norm.normalize(goals))

        actions = self.online_policy_network.sample(observations, goals, noise=noise, evaluate=evaluate)

        if self.mode == 'single_env':
            actions = actions.squeeze()

        return actions.data.numpy()

    def train(self, batch):
        obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = batch
        # normalize actions before Q network
        action_scale = self.policy_network.action_scale
        action_bias = self.policy_network.action_bias
        actions = (actions - action_bias) / action_scale
        # normalize observations and goals
        obs = self.obs_norm.normalize(obs, device=self.device)
        des_goals = self.goal_norm.normalize(des_goals, device=self.device)
        # ach_goals = self.goal_norm.normalize(ach_goals, device=self.device)
        next_obs = self.obs_norm.normalize(next_obs, device=self.device)
        next_des_goals = self.goal_norm.normalize(next_des_goals, device=self.device)
        next_ach_goals = self.goal_norm.normalize(next_ach_goals, device=self.device)

        goals = des_goals  # - ach_goals
        next_goals = next_des_goals  # - next_ach_goals

        # Q network update
        self.q_opt.zero_grad()

        with torch.no_grad():
            next_actions = self.target_policy_network.sample(next_obs, next_goals, noise=False, evaluate=False)
            next_actions = (next_actions - action_bias) / action_scale
            target_q_next = self.target_q_network(next_obs, next_goals, next_actions)

        q_hat = rewards + (1 - dones) * self.gamma * target_q_next.detach()

        q_loss = torch.mean(0.5 * (self.q_network(obs, goals, actions) - q_hat) ** 2)
        q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=0.1)
        self.q_opt.step()

        # Policy network update
        policy_actions = self.policy_network.sample(obs, goals, noise=False, evaluate=False)
        norm_policy_actions = (policy_actions - action_bias) / action_scale
        q = self.q_network(obs, goals, norm_policy_actions)

        policy_loss = -torch.mean(q) + torch.mean(policy_actions / action_scale).pow(2)
        # if epoch < 10000:
        #    policy_loss -= policy_loss.cpu().data.numpy()*0.06*torch.mean(0.5*policy_actions.pow(2))

        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.1)
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        self._soft_update(self.target_q_network, self.q_network, self.tau)
        self._soft_update(self.target_policy_network, self.policy_network, self.tau)

        # check online cpu policy network location and copy trained weights into it to play in the environment
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())
        assert self.online_policy_network.layer2.weight.device == torch.device('cpu'), \
            "Online network was placed on cuda or somewhere else. Please check the location!"

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
        self.goal_norm.save(f'./weights/{model_name}', 'goal')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/policy_network.pt')
        torch.save(self.q_network.state_dict(), f'./weights/{model_name}/q_network.pt')
        torch.save(self.target_q_network.state_dict(), f'./weights/{model_name}/target_q_network.pt')
        torch.save(self.target_policy_network.state_dict(), f'./weights/{model_name}/target_policy_network.pt')

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
            self.online_policy_network.load_state_dict(torch.load(f"./weights/{model_name}/policy_network.pt",
                                                       map_location=lambda storage, loc: storage))
            return

        self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}/policy_network.pt",
                                                       map_location=lambda storage, loc: storage))
        self.q_network.load_state_dict(torch.load(f"./weights/{model_name}/q_network.pt",
                                                  map_location=lambda storage, loc: storage))
        self.target_q_network.load_state_dict(torch.load(f"./weights/{model_name}/target_q_network.pt",
                                                         map_location=lambda storage, loc: storage))
        self.target_policy_network.load_state_dict(torch.load(f"./weights/{model_name}/target_policy_network.pt",
                                                              map_location=lambda storage, loc: storage))
