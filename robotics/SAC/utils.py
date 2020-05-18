import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from collections import namedtuple
import cloudpickle
from matplotlib import pyplot as plt
import os

MIN_LOG_SIGMA = -20
MAX_LOG_SIGMA = 2
eps = 1e-6


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=2, padding=2)  # 128
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=2, padding=1)  # 64
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)  # 32
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)  # 16
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=1, padding=0)  # 8
        self.conv7 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)  # (8, 8, 32)
        self.flatten = Flatten()  # 8*8*32 = 2048

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = F.leaky_relu(self.conv5(out))
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))
        out = self.flatten(out)

        return out


class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, goal_shape, output_shape, action_ranges, include_conv=True):
        super(PolicyNetwork, self).__init__()
        self.action_ranges = action_ranges
        self.include_conv = include_conv
        if include_conv:
            self.conv_layers = ConvModule()

        self.layer_obs = nn.Linear(2048 if include_conv else observation_shape, 200)
        self.layer_goal = nn.Linear(goal_shape, 200)
        self.layer1 = nn.Linear(400, 100)
        self.layer2 = nn.Linear(100, output_shape)
        self.layer3 = nn.Linear(100, output_shape)

        self.action_scale = (action_ranges[1] - action_ranges[0]) / 2
        self.action_bias = (action_ranges[1] + action_ranges[0]) / 2

    def forward(self, observation, goals):
        if self.include_conv:
            observation = self.conv_layers(observation)
        processed_obs = F.relu(self.layer_obs(observation))
        processed_goal = F.relu(self.layer_goal(goals))
        if len(processed_goal.shape) < len(processed_obs.shape):
            processed_goal = processed_goal[np.newaxis, :]

        out = torch.cat([processed_obs, processed_goal], dim=-1)
        out = F.relu(self.layer1(out))

        mu = self.layer2(out)
        log_sigma = torch.relu(self.layer3(out))
        log_sigma = torch.clamp(log_sigma, MIN_LOG_SIGMA, MAX_LOG_SIGMA)

        return mu, log_sigma

    def sample(self, observations, goals, evaluate=False):
        mu, log_sigma = self.forward(observations, goals)

        if evaluate:
            return torch.tanh(mu) * self.action_scale + self.action_bias

        sigma = log_sigma.exp()
        distribution = Normal(mu, sigma)
        x_t = distribution.rsample()
        # action = torch.clamp(action, self.action_ranges[0], self.action_ranges[1])
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(x_t)
        # from SAC paper original, page 12
        log_prob -= torch.log(self.action_scale*(1 - y_t.pow(2)) + eps)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, observation_shape, goal_shape, action_shape, include_conv=True):
        super(QNetwork, self).__init__()
        self.include_conv = include_conv
        if include_conv:
            self.conv_layers = ConvModule()

        self.layer_obs = nn.Linear(2048 if include_conv else observation_shape, 256)
        self.layer_goal = nn.Linear(goal_shape, 256)
        self.layer1 = nn.Linear(512, 256)

        self.layer_action = nn.Linear(action_shape, 256)

        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, observation, goal, action):
        if self.include_conv:
            observation = self.conv_layers(observation)
        obs = F.leaky_relu(self.layer_obs(observation))
        goal_ = F.leaky_relu(self.layer_goal(goal))
        obs_goal = torch.cat([obs, goal_], dim=-1)

        state = F.leaky_relu(self.layer1(obs_goal))

        action_ = F.leaky_relu(self.layer_action(action))

        state_action = torch.cat([state, action_], dim=-1)
        out = F.leaky_relu(self.layer2(state_action))
        out = F.leaky_relu(self.layer3(out))

        return out


item = namedtuple("experience_replay_item", ("ob", "des_goal", "ach_goal", "action",
                                             "reward", "next_ob", "next_des_goal",
                                             "next_ach_goal", "done"))


class ExperienceReplay:
    def __init__(self, size=100000, mode='multi_env', device='cuda'):
        self.size = size
        self.data = []
        self._next = 0
        self.mode = mode
        self.device = device

    def put(self, states, actions, rewards, next_states, dones):
        obs = states["observation"]
        des_goals = states["desired_goal"]
        ach_goals = states["achieved_goal"]

        next_obs = next_states["observation"]
        next_des_goals = next_states["desired_goal"]
        next_ach_goals = next_states["achieved_goal"]

        if self.mode == 'single_env':
            ob = torch.FloatTensor(obs)
            des_goal = torch.FloatTensor(des_goals)
            ach_goal = torch.FloatTensor(ach_goals)
            action = torch.FloatTensor(actions)
            next_ob = torch.FloatTensor(next_obs)
            next_des_goal = torch.FloatTensor(next_des_goals)
            next_ach_goal = torch.FloatTensor(next_ach_goals)
            done = dones
            reward = rewards
            transition = self._get_transition(ob, des_goal, ach_goal, action, reward, next_ob,
                                              next_des_goal, next_ach_goal, done)
            if self._next >= len(self.data):
                self.data.append(transition)
            else:
                self.data[self._next] = transition

            self._next = (self._next + 1) % self.size

            return
        else:
            for i in range(obs.shape[0]):
                ob = torch.FloatTensor(obs[i])
                des_goal = torch.FloatTensor(des_goals[i])
                ach_goal = torch.FloatTensor(ach_goals[i])
                action = torch.FloatTensor(actions[i])
                next_ob = torch.FloatTensor(next_obs[i])
                next_des_goal = torch.FloatTensor(next_des_goals[i])
                next_ach_goal = torch.FloatTensor(next_ach_goals[i])
                done = dones[i]
                reward = rewards[i]

                transition = self._get_transition(ob, des_goal, ach_goal, action, reward, next_ob,
                                                  next_des_goal, next_ach_goal, done)

                if self._next >= len(self.data):
                    self.data.append(transition)
                else:
                    self.data[self._next] = transition

                self._next = (self._next + 1) % self.size

    @staticmethod
    def _get_transition(ob, des_goal, ach_goal, action, reward, next_ob, next_des_goal, next_ach_goal, done):
        return item(ob, des_goal, ach_goal, action, reward, next_ob, next_des_goal, next_ach_goal, done)

    def save(self, filename):
        with open(filename, 'wb') as f:
            cloudpickle.dump(self.data, f)

    def sample(self, batch_size):
        O, G_d, G_a = [], [], []
        A = []
        R = []
        O_n, G_d_n, G_a_n = [], [], []
        dones = []
        idxs = np.random.choice(len(self.data), batch_size, replace=False)
        for idx in idxs:
            sample = self.data[idx]
            O.append(sample.ob)
            G_d.append(sample.des_goal)
            G_a.append(sample.ach_goal)
            A.append(sample.action)
            R.append(sample.reward)
            O_n.append(sample.next_ob)
            G_d_n.append(sample.next_des_goal)
            G_a_n.append(sample.next_ach_goal)
            dones.append(sample.done)

        O = torch.stack(O).to(self.device)
        G_d = torch.stack(G_d).to(self.device)
        G_a = torch.stack(G_a).to(self.device)
        A = torch.stack(A).to(self.device)
        R = torch.FloatTensor(R).to(self.device)
        O_n = torch.stack(O_n).to(self.device)
        G_d_n = torch.stack(G_d_n).to(self.device)
        G_a_n = torch.stack(G_a_n).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return O, G_d, G_a, A, R[:, np.newaxis], O_n, G_d_n, G_a_n, dones[:, np.newaxis]

    def __len__(self):
        return len(self.data)


class HindsightExperienceReplay:
    def __init__(self, env_params, n_envs, k=8, size=20000, use_achieved_goal=False, device='cuda'):
        self.env_params = env_params
        self.size = size
        self.data = {"obs": np.empty([size, env_params["max_episode_timesteps"], env_params['obs']]),
                     "actions": np.empty([size, env_params["max_episode_timesteps"], env_params['actions']]),
                     "goals": np.empty([size, env_params["max_episode_timesteps"], env_params['goals']]),
                     "ach_goals": np.empty([size, env_params["max_episode_timesteps"], env_params['goals']]),
                     "next_obs": np.empty([size, env_params["max_episode_timesteps"], env_params['obs']]),
                     "next_ach_goals": np.empty([size, env_params["max_episode_timesteps"], env_params['goals']]),
                     "rewards": np.empty([size, env_params["max_episode_timesteps"], 1]),
                     "dones": np.empty([size, env_params["max_episode_timesteps"], 1])
                     }
        self._next = 0
        self.n_envs = n_envs
        self.k = k
        self.reward_function = env_params['reward_function']
        self.device = device
        self.current_size = 0
        self.use_achieved_goal = use_achieved_goal

        self.episode_data = self._get_episode_data()

    def _get_episode_data(self):
        return {"obs": [[] for _ in range(self.n_envs)],
                "actions": [[] for _ in range(self.n_envs)],
                "goals": [[] for _ in range(self.n_envs)],
                "ach_goals": [[] for _ in range(self.n_envs)],
                "next_obs": [[] for _ in range(self.n_envs)],
                "next_ach_goals": [[] for _ in range(self.n_envs)],
                "rewards": [[] for _ in range(self.n_envs)],
                "dones": [[] for _ in range(self.n_envs)]
                }

    def collect_episodes(self, states, actions, rewards, next_states, dones):
        obs = states["observation"]
        goals = states["desired_goal"]
        ach_goals = states["achieved_goal"]
        next_obs = next_states["observation"]
        next_ach_goals = next_states["achieved_goal"]

        for n in range(self.n_envs):
            self.episode_data["obs"][n].append(obs[n])
            self.episode_data["goals"][n].append(goals[n])
            self.episode_data["ach_goals"][n].append(ach_goals[n])
            self.episode_data["actions"][n].append(actions[n])
            self.episode_data["rewards"][n].append([rewards[n]])
            self.episode_data["next_obs"][n].append(next_obs[n])
            self.episode_data["next_ach_goals"][n].append(next_ach_goals[n])
            self.episode_data["dones"][n].append([dones[n]])

    def store_episodes(self):
        obs = np.array(self.episode_data['obs'])
        goals = np.array(self.episode_data['goals'])
        ach_goals = np.array(self.episode_data['ach_goals'])
        actions = np.array(self.episode_data['actions'])
        rewards = np.array(self.episode_data['rewards'])
        next_obs = np.array(self.episode_data['next_obs'])
        next_ach_goals = np.array(self.episode_data['next_ach_goals'])
        dones = np.array(self.episode_data['dones'])

        idx = self._get_idx()
        self.data['obs'][idx] = obs
        self.data['goals'][idx] = goals
        self.data['ach_goals'][idx] = ach_goals
        self.data['actions'][idx] = actions
        self.data['rewards'][idx] = rewards
        self.data['next_obs'][idx] = next_obs
        self.data['next_ach_goals'][idx] = next_ach_goals
        self.data['dones'][idx] = dones

        self.episode_data = self._get_episode_data()

    def _get_idx(self):
        if self.current_size + self.n_envs < self.size:
            idx = np.arange(self.current_size, self.current_size + self.n_envs)
        elif self.current_size < self.size:
            overflow = self.n_envs - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, self.n_envs)

        self.current_size = min(self.size, self.current_size + self.n_envs)

        return idx

    def her_sample(self, batch_size):
        transitions = {key: self.data[key][:self.current_size, 1:, :].copy() for key in self.data.keys()}
        # TODO: maybe it is more useful to omit next_obs and next goals (next achieved goals) from self.data
        episode_length = transitions['dones'].shape[1]
        number_of_episodes_in_buffer = transitions['dones'].shape[0]

        # Select which episodes and time steps to use.
        # (Nikita) Choose episodes indices, which will be sample as transitions for training
        episode_idxs = np.random.randint(0, number_of_episodes_in_buffer, size=batch_size)
        # (Nikita) Choose episode timesteps which will be changed
        t_samples = np.random.randint(episode_length, size=batch_size)
        # (Nikita) Get transitions that was chosen in previous lines
        transitions = {key: transitions[key][episode_idxs, t_samples].copy() for key in transitions}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        # (Nikita) Get exactly transitions indices where substitute goal and achieved goal
        her_indices = np.where(np.random.uniform(size=batch_size) < 1 - (1. / (1 + self.k)))
        future_offset = (np.random.uniform(size=batch_size) * (episode_length - t_samples)).astype('int')
        future_t = (t_samples + 1 + future_offset)[her_indices]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_achieved_goal = np.array(self.data['ach_goals'])[episode_idxs[her_indices], future_t]
        transitions['goals'][her_indices] = future_achieved_goal
        # transitions['dones'][her_indices] = np.ones_like(transitions['dones'][her_indices]).astype('bool')

        # Recompute rewards since we may have substituted the goal.
        transitions['rewards'] = self.reward_function(transitions['next_ach_goals'],
                                                      transitions['goals'], None)[:, np.newaxis]

        return transitions

    def _sample(self, batch_size):
        # Standard experience replay sampling
        episode_idx = np.random.randint(0, self.current_size, batch_size)
        timestep_idx = np.random.randint(0, self.env_params['max_episode_timesteps'], batch_size)
        transitions = {key: self.data[key][episode_idx, timestep_idx] for key in self.data.keys()}

        return transitions

    def sample(self, batch_size):
        transitions = self.her_sample(batch_size)
        # ordinary_transitions = self._sample(batch_size)
        # transitions = {key: np.concatenate([her_transitions[key], ordinary_transitions[key]])
        # for key in her_transitions.keys()}

        obs = torch.FloatTensor(transitions['obs']).to(self.device)
        goals = transitions['goals'] - transitions['ach_goals'] if self.use_achieved_goal else transitions['goals']
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(transitions['actions']).to(self.device)
        rewards = torch.FloatTensor(transitions['rewards']).to(self.device)
        next_obs = torch.FloatTensor(transitions['next_obs']).to(self.device)
        next_goals = transitions['goals'] - transitions['next_ach_goals'] if self.use_achieved_goal \
            else transitions['goals']
        next_goals = torch.FloatTensor(next_goals).to(self.device)
        dones = torch.FloatTensor(transitions['dones']).to('cuda')

        return obs, goals, actions, rewards, next_obs, next_goals, dones


class Normalizer:
    def __init__(self, size, multi_env=True):
        self.size = size
        self.mean = np.zeros(size, dtype='float32')
        self.std = np.ones(size, dtype='float32')
        self.m = 0
        self.multi_env = multi_env

    def normalize(self, inputs, device='cpu'):
        mean = self.mean if device == 'cpu' else torch.FloatTensor(self.mean).to(device)
        std = self.std if device == 'cpu' else torch.FloatTensor(self.std).to(device)
        return (inputs - mean) / std

    def update_stats(self, inputs):
        # inputs.shape = [batch_size, obs_shape]
        if self.multi_env:
            assert len(inputs.shape) == 2
            assert inputs.shape[1] == self.size
        n = inputs.shape[0]
        inputs_mean = np.mean(inputs, axis=0)
        inputs_std = np.std(inputs, axis=0)

        self.std = np.sqrt((self.m*self.std**2 + n*inputs_std**2) / (n + self.m) +
                           n*self.m*((self.mean - inputs_mean)**2) / (n + self.m)**2) + 1.01e-6
        self.mean = (self.m*self.mean + n*inputs_mean) / (n + self.m)

        self.m += n

        assert self.std.all() > 1e-6, print(self.std, inputs_std, inputs)

    def save(self, path, name):
        np.save(f'{path}/{name}_norm_mean', self.mean)
        np.save(f'{path}/{name}_norm_std', self.std)

    def load(self, path, name):
        self.mean = np.load(f'{path}/{name}_norm_mean.npy')
        self.std = np.load(f'{path}/{name}_norm_std.npy')


