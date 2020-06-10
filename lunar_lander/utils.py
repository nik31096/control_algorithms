import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from collections import namedtuple
import cloudpickle

MIN_LOG_SIGMA = -20
MAX_LOG_SIGMA = 2
eps = 1e-6


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, output_shape, action_ranges):
        super(PolicyNetwork, self).__init__()
        self.action_ranges = action_ranges

        self.layer1 = nn.Linear(observation_shape, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_shape)

        self.action_scale = (action_ranges[1] - action_ranges[0]) / 2
        self.action_bias = (action_ranges[1] + action_ranges[0]) / 2

        self.noise = Normal(0, 2*self.action_scale)

    def forward(self, observation):
        out = F.relu(self.layer1(observation))
        out = F.relu(self.layer2(out))
        action = self.layer3(out)

        return action

    def sample(self, observations, add_noise=True, evaluate=False):
        action = self.forward(observations)

        if add_noise:
            action += self.noise.sample(sample_shape=action.shape).to(action.device)
            action = torch.tanh(action) * self.action_scale + self.action_bias
        elif evaluate:
            action = torch.tanh(action) * self.action_scale + self.action_bias
        else:
            action = torch.tanh(action)

        return action


class PolicyNetwork_SAC(nn.Module):
    def __init__(self, observation_shape, goal_shape, output_shape, action_ranges):
        super(PolicyNetwork_SAC, self).__init__()
        self.action_ranges = action_ranges

        self.layer_obs = nn.Linear(observation_shape, 200)
        self.layer_goal = nn.Linear(goal_shape, 200)
        self.layer1 = nn.Linear(400, 100)
        self.layer2 = nn.Linear(100, output_shape)
        self.layer3 = nn.Linear(100, output_shape)

        self.action_scale = (action_ranges[1] - action_ranges[0]) / 2
        self.action_bias = (action_ranges[1] + action_ranges[0]) / 2

    def forward(self, observation):
        if self.include_conv:
            observation = self.conv_layers(observation)
        out = F.relu(self.layer_obs(observation))

        out = F.relu(self.layer1(out))

        mu = self.layer2(out)
        log_sigma = torch.relu(self.layer3(out))
        log_sigma = torch.clamp(log_sigma, MIN_LOG_SIGMA, MAX_LOG_SIGMA)

        return mu, log_sigma

    def sample(self, observations, evaluate=False):
        mu, log_sigma = self.forward(observations)

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
    def __init__(self, observation_shape, action_shape):
        super(QNetwork, self).__init__()

        self.layer_obs = nn.Linear(observation_shape, 256)
        self.layer1 = nn.Linear(256, 256)

        self.layer_action = nn.Linear(action_shape, 256)

        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, observation, action):
        obs = F.leaky_relu(self.layer_obs(observation))

        action_ = F.leaky_relu(self.layer_action(action))

        state_action = torch.cat([obs, action_], dim=-1)
        out = F.leaky_relu(self.layer2(state_action))
        out = F.leaky_relu(self.layer3(out))

        return out


class QNetwork_NAF(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(QNetwork_NAF, self).__init__()

        self.layer_obs = nn.Linear(observation_shape, 256)

        self.layer_action = nn.Linear(action_shape, 256)

        self.layer = nn.Linear(512, 256)
        self.V = nn.Linear(256, 1)
        self.mu = nn.Linear(256, action_shape)
        self.L = nn.Linear(256, action_shape*action_shape)

    def forward(self, observation, action):
        obs = F.leaky_relu(self.layer_obs(observation))

        action_ = F.leaky_relu(self.layer_action(action))

        state_action = torch.cat([obs, action_], dim=-1)
        out = F.leaky_relu(self.layer(state_action))

        V = self.V(out)
        mu = torch.tanh(self.mu(out))
        L = self.L(out).view(-1, action_.shape[-1], action_.shape[-1])
        L = L*torch.tril(torch.ones(action_.shape[-1], action_.shape[-1]))[np.newaxis, :, :]
        assert len(L.shape) == 3, L.shape
        P = torch.bmm(L, L.transpose(2, 1))
        A = -0.5*torch.bmm((action - mu).T, torch.bmm(P, action - mu))
        assert A.shape[-1] == 1, A.shape
        Q = A - V

        return Q


item = namedtuple("experience_replay_item", ("ob", "action", "reward", "next_ob", "done"))


class ExperienceReplay:
    def __init__(self, size=100000, mode='multi_env', device='cuda'):
        self.size = size
        self.data = []
        self._next = 0
        self.mode = mode
        self.device = device

    def put(self, states, actions, rewards, next_states, dones):

        if self.mode == 'single_env':
            ob = torch.FloatTensor(states)
            action = torch.FloatTensor(actions)
            next_ob = torch.FloatTensor(next_states)
            done = dones
            reward = rewards
            transition = self._get_transition(ob, action, reward, next_ob, done)
            if self._next >= len(self.data):
                self.data.append(transition)
            else:
                self.data[self._next] = transition

            self._next = (self._next + 1) % self.size

            return
        else:
            for i in range(states.shape[0]):
                ob = torch.FloatTensor(states[i])
                action = torch.FloatTensor(actions[i])
                next_ob = torch.FloatTensor(next_states[i])
                done = dones[i]
                reward = rewards[i]

                transition = self._get_transition(ob, action, reward, next_ob, done)

                if self._next >= len(self.data):
                    self.data.append(transition)
                else:
                    self.data[self._next] = transition

                self._next = (self._next + 1) % self.size

    @staticmethod
    def _get_transition(ob, action, reward, next_ob, done):
        return item(ob, action, reward, next_ob, done)

    def save(self, filename):
        with open(filename, 'wb') as f:
            cloudpickle.dump(self.data, f)

    def sample(self, batch_size):
        O = []
        A = []
        R = []
        O_n = []
        dones = []
        idxs = np.random.choice(len(self.data), batch_size, replace=False)
        for idx in idxs:
            sample = self.data[idx]
            O.append(sample.ob)
            A.append(sample.action)
            R.append(sample.reward)
            O_n.append(sample.next_ob)
            dones.append(sample.done)

        O = torch.stack(O).to(self.device)
        A = torch.stack(A).to(self.device)
        R = torch.FloatTensor(R).to(self.device)
        O_n = torch.stack(O_n).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return O, A, R[:, np.newaxis], O_n, dones[:, np.newaxis]

    def __len__(self):
        return len(self.data)


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
