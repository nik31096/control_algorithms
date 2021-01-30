import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from collections import namedtuple
import cloudpickle
from matplotlib import pyplot as plt
import os


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
    def __init__(self, observation_shape, goal_shape, output_shape, include_conv=True):
        super(PolicyNetwork, self).__init__()
        self.include_conv = include_conv
        if include_conv:
            self.conv_layers = ConvModule()

        self.layer_obs = nn.Linear(2048 if include_conv else observation_shape, 200)
        self.layer_goal = nn.Linear(goal_shape, 200)
        self.layer1 = nn.Linear(400, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_shape)

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
        action = torch.tanh(self.layer3(out))

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
