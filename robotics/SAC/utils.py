import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

MIN_LOG_SIGMA = -20
MAX_LOG_SIGMA = 2
eps = 1e-6


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, goal_shape, output_shape, action_ranges):
        super(PolicyNetwork, self).__init__()
        self.action_ranges = action_ranges

        self.layer_obs = nn.Linear(observation_shape, 200)
        self.layer_goal = nn.Linear(goal_shape, 200)
        self.layer1 = nn.Linear(400, 100)
        self.layer2 = nn.Linear(100, output_shape)
        self.layer3 = nn.Linear(100, output_shape)

        self.action_scale = (action_ranges[1] - action_ranges[0]) / 2
        self.action_bias = (action_ranges[1] + action_ranges[0]) / 2

    def forward(self, observation, goals):
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
    def __init__(self, observation_shape, goal_shape, action_shape):
        super(QNetwork, self).__init__()

        self.layer_obs = nn.Linear(observation_shape, 256)
        self.layer_goal = nn.Linear(goal_shape, 256)
        self.layer1 = nn.Linear(512, 256)

        self.layer_action = nn.Linear(action_shape, 256)

        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, observation, goal, action):
        obs = F.leaky_relu(self.layer_obs(observation))
        goal_ = F.leaky_relu(self.layer_goal(goal))
        obs_goal = torch.cat([obs, goal_], dim=-1)

        state = F.leaky_relu(self.layer1(obs_goal))

        action_ = F.leaky_relu(self.layer_action(action))

        state_action = torch.cat([state, action_], dim=-1)
        out = F.leaky_relu(self.layer2(state_action))
        out = F.leaky_relu(self.layer3(out))

        return out


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


