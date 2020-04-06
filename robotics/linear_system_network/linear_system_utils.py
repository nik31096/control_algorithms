import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
import numpy as np


item = namedtuple(
    "Item",
    ["obs", "pos", "action", "next_pos"]
)


class ExperienceReplay:
    def __init__(self, size=100000, cuda=True):
        self.size = size
        self.data = []
        self._next = 0
        self.cuda = cuda

    def put(self, states, actions, next_states):
        '''
        :param states: big dictionary with keys: "observation", "desired_goal", "achieved_goal"
        :param actions: 2d array of shape (n_envs, action_dim)
        :param next_states: same as states
        '''

        obs = states['observation']
        positions = states['achieved_goal']
        next_positions = next_states['achieved_goal']
        for i in range(obs.shape[0]):
            ob = torch.FloatTensor(obs[i])
            pos = torch.FloatTensor(positions[i])
            next_pos = torch.FloatTensor(next_positions[i])
            action = torch.FloatTensor(actions[i])

            if self._next >= len(self.data):
                self.data.append(item(ob, pos, action, next_pos))
            else:
                self.data[self._next] = item(ob, pos, action, next_pos)

            self._next = (self._next + 1) % self.size

    def sample(self, batch_size):
        obs_pos, pos, obs_actions, actions, next_pos = [], [], [], [], []
        idxs = np.random.randint(0, len(self.data), batch_size)
        for idx in idxs:
            sample = self.data[idx]
            obs_pos.append(torch.cat([sample.obs, sample.pos], dim=-1))
            pos.append(sample.pos)
            obs_actions.append(torch.cat([sample.obs, sample.action], dim=-1))
            actions.append(sample.action)
            next_pos.append(sample.next_pos)

        if self.cuda:
            return torch.stack(obs_pos).to('cuda'), torch.stack(pos).to('cuda'), torch.stack(obs_actions).to('cuda'), \
                   torch.stack(actions).to('cuda'), torch.stack(next_pos).to('cuda')
        else:
            return torch.stack(obs_pos), torch.stack(pos), torch.stack(obs_actions), torch.stack(actions), \
                   torch.stack(next_pos)

    def __len__(self):
        return len(self.data)


def swish(x):
    return x * torch.sigmoid(x)


class Network(nn.Module):
    def __init__(self, input_dim, output_shape):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_shape[0] * output_shape[1])
        self.output_shape = output_shape

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x))
        out = F.leaky_relu(self.layer2(out))
        out = swish(self.layer3(out))
        out = torch.reshape(out, (x.shape[0], *self.output_shape))

        return out