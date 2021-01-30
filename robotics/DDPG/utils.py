import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from collections import namedtuple
import cloudpickle


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
        self.layer1 = nn.Linear(400, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_shape)

        self.action_scale = (action_ranges[1] - action_ranges[0]) / 2
        self.action_bias = (action_ranges[1] + action_ranges[0]) / 2

        self.noise = Normal(0, 3*self.action_scale)

    def forward(self, observation, goals):
        if self.include_conv:
            observation = self.conv_layers(observation)
        processed_obs = F.relu(self.layer_obs(observation))
        processed_goal = F.relu(self.layer_goal(goals))
        if len(processed_goal.shape) < len(processed_obs.shape):
            processed_goal = processed_goal[np.newaxis, :]

        out = torch.cat([processed_obs, processed_goal], dim=-1)
        out = F.relu(self.layer1(out))
        out = F.leaky_relu(self.layer2(out))
        action = self.layer3(out)

        return action

    def sample(self, observations, goals, noise=True, evaluate=False):
        action = self.forward(observations, goals)

        if noise:
            action += self.noise.sample(sample_shape=action.shape).to(action.device)
            action = torch.tanh(action) * self.action_scale + self.action_bias
        elif evaluate:
            action = torch.tanh(action) * self.action_scale + self.action_bias
        else:
            action = torch.tanh(action)

        return action


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


class BaselineNormilizer:
    """
    Code was taken from https://github.com/TianhongDai/hindsight-experience-replay/blob/master/mpi_utils/normalizer.py
    """
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
