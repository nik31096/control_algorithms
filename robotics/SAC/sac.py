import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, goal_shape, action_shape, action_ranges):
        self.action_ranges = action_ranges
        self.layer_obs = nn.Linear(observation_shape, 256)
        self.layer_goal = nn.Linear(goal_shape, 256)
        self.layer1 = nn.Linear(512, 256)
        self.layer_mu = nn.Linear(256, action_shape)
        self.layer_sigma = nn.Linear(256, action_shape)

    def forward(self, obs, goal):
        obs_ = F.relu(self.layer_obs(obs))
        goal_ = F.relu(self.layer_goal(goal))
        out = torch.stack([obs_, goal_], dim=-1)
        out = F.relu(self.layer1(out))

        mu = F.tanh(self.layer_mu(out))
        sigma = F.relu(self.layer_sigma(out))

        distribution = torch.distributions.Normal(mu, sigma)
        action = distribution.sample() + torch.randn_like(mu)*0.1

        return torch.clamp(action, self.action_ranges[0], self.action_ranges[1])
    
    def entropy(self, obs):
        pass


class ValueNetwork(nn.Module):
    def __init__(self, observation_shape, goal_shape):
        self.layer1 = nn.Linear(observation_shape, 256)
        self.layer2 = nn.Linear(goal_shape, 256)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, obs, goal):
        out1 = F.leaky_relu(self.layer1(obs))
        out2 = F.leaky_relu(self.layer2(goal))
        out = torch.stack([out1, out2], dim=-1)
        out = F.leaky_relu(self.layer3(out))
        out = F.leaky_relu(self.layer4(out))

        return out





# In[5]:


class ExperienceReplay:
    def __init__(self, size):
        self.data = []
        self.size = size
        self._next = 0
        
    def put(self, item):
        if self_next >= len(self.data):
            self.data.append(item)
        else:
            self.data[self._next] = item
            
        self._next = (self._next + 1) % self.size
    
    def __len__(self):
        return len(self.data)

    def sample(self, batch_size):
        O, G_d, G_a, A, R, O_1, G_d_1, G_a_1, D = [], [], [], [], [], [], [], [], []
        idxs = np.random.choice(0, len(self.data), batch_size)
        for i in idxs:
            item = self.data[i]
            O.append(item.obs)
            G_a.append(item.desired_goal)
            G_d.append(item.achieved_goal)
            A.append(item.action)
            R.append(item.reward)
            O_1.append(item.next_obs)
            G_d_1.append(item.next_desired_goal)
            G_a_1.append(item.next_achieved_goal)
            D.append(item.done)
            
        O = torch.stack(O)
        G_d = torch.stack(G_d)
        G_a = torch.stack(G_a)
        A = torch.stack(A)
        R = torch.stack(R)
        O_1 = torch.stack(O_1)
        G_d_1 = torch.stack(G_d_1)
        G_a_1 = torch.stack(G_a_1)
        D = torch.stack(D)
        
        return O, G_d, G_a, A, R, O_1, G_d_1, G_a_1, D



