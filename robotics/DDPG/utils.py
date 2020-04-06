import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import namedtuple
import random
import cloudpickle


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, goal_shape, output_shape, action_ranges):
        super(PolicyNetwork, self).__init__()
        self.action_ranges = action_ranges
        self.layer_obs = nn.Linear(obs_shape, 200)
        self.layer_goal = nn.Linear(2 * goal_shape, 200)
        self.layer1 = nn.Linear(400, 100)
        self.layer2 = nn.Linear(100, output_shape)

    def forward(self, observation, desired_goal, achieved_goal):
        # check if observation, desired_goal and achieved_goal are torch tensors
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation)
            desired_goal = torch.FloatTensor(desired_goal)
            achieved_goal = torch.FloatTensor(achieved_goal)

        processed_obs = F.leaky_relu(self.layer_obs(observation))
        concat_goal = torch.cat([desired_goal, achieved_goal], dim=-1)
        processed_goal = F.leaky_relu(self.layer_goal(concat_goal))

        out = torch.cat([processed_obs, processed_goal], dim=-1)
        out = F.leaky_relu(self.layer1(out))

        mu = torch.tanh(self.layer2(out))

        # TODO: think about noise added to action to improve exploration. Do we really need this?
        action = mu + torch.randn_like(mu) * 0.1
        action = torch.clamp(action, self.action_ranges[0], self.action_ranges[1])

        return action


class ValueNetwork(nn.Module):
    def __init__(self, obs_shape, goal_shape, action_shape):
        super(ValueNetwork, self).__init__()
        self.layer_obs = nn.Linear(obs_shape, 200)
        self.layer_goal = nn.Linear(2 * goal_shape, 200)
        self.layer_action = nn.Linear(action_shape, 200)
        self.layer1 = nn.Linear(400, 200)
        self.layer2 = nn.Linear(400, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, observations, desired_goals, achieved_goals, actions):
        processed_obs = F.leaky_relu(self.layer_obs(observations))
        concat_goal = torch.cat([desired_goals, achieved_goals], dim=-1)
        processed_goals = F.leaky_relu(self.layer_goal(concat_goal))
        out = torch.cat([processed_obs, processed_goals], dim=-1)
        out = F.leaky_relu(self.layer1(out))

        processed_actions = F.leaky_relu(self.layer_action(actions))
        out = torch.cat([out, processed_actions], dim=-1)
        out = F.leaky_relu(self.layer2(out))
        out = F.leaky_relu(self.layer3(out))

        return out


item = namedtuple(
    "Item",
    ["obs", "des_goal", "ach_goal", "action", "reward", "next_obs", "next_des_goal", "next_ach_goal", "done"]
)


class ExperienceReplay:
    def __init__(self, size=100000, mode='multi_env'):
        self.size = size
        self.data = []
        self._next = 0
        self.mode = mode

    def put(self, states, actions, rewards, next_states, dones):
        if self.mode == 'multi_env':
            obs = states["observation"]
            des_goals = states["desired_goal"]
            ach_goals = states["achieved_goal"]

            next_obs = next_states["observation"]
            next_des_goals = next_states["desired_goal"]
            next_ach_goals = next_states["achieved_goal"]

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

                if reward == -1.0 and done is True and random.random() < 0.2:
                    reward = 1

                if self._next >= len(self.data):
                    self.data.append(
                        item(ob, des_goal, ach_goal, action, reward, next_ob, next_des_goal, next_ach_goal, done)
                    )
                else:
                    self.data[self._next] = item(ob, des_goal, ach_goal, action, -0.1,
                                                 next_ob, next_des_goal, next_ach_goal, done)
                self._next = (self._next + 1) % self.size
        else:
            # rename variables
            state = states
            next_state = next_states
            action = actions
            reward = rewards
            done = dones

            ob = torch.FloatTensor(state['observation'])
            des_goal = torch.FloatTensor(state['desired_goal'])
            ach_goal = torch.FloatTensor(state['achieved_goal'])
            next_ob = torch.FloatTensor(next_state['observation'])
            next_des_goal = torch.FloatTensor(state['desired_goal'])
            next_ach_goal = torch.FloatTensor(state['achieved_goal'])

            if reward == -1.0 and done is True and random.random() < 0.2:
                reward = 1

            if self._next >= len(self.data):
                self.data.append(
                    item(ob, des_goal, ach_goal, action, reward, next_ob, next_des_goal, next_ach_goal, done)
                )
            else:
                self.data[self._next] = item(ob, des_goal, ach_goal, action, -0.1,
                                             next_ob, next_des_goal, next_ach_goal, done)

            self._next = (self._next + 1) % self.size

    def save(self, filename):
        with open(filename, 'wb') as f:
            cloudpickle.dump(self.data, f)

    def sample(self, batch_size):
        if self.mode == 'multi_env':
            O, G_d, G_a = [], [], []
            A = []
            R = []
            O_n, G_d_n, G_a_n = [], [], []
            dones = []
            idxs = np.random.choice(len(self.data), batch_size, replace=False)
            for idx in idxs:
                sample = self.data[idx]
                O.append(sample.obs)
                G_d.append(sample.des_goal)
                G_a.append(sample.ach_goal)
                A.append(sample.action)
                R.append(sample.reward)
                O_n.append(sample.next_obs)
                G_d_n.append(sample.next_des_goal)
                G_a_n.append(sample.next_ach_goal)
                dones.append(sample.done)

            O = torch.stack(O).to('cuda')
            G_d = torch.stack(G_d).to('cuda')
            G_a = torch.stack(G_a).to('cuda')
            A = torch.stack(A).to('cuda')
            R = torch.FloatTensor(R).to('cuda')
            O_n = torch.stack(O_n).to('cuda')
            G_d_n = torch.stack(G_d_n).to('cuda')
            G_a_n = torch.stack(G_a_n).to('cuda')
            dones = torch.FloatTensor(dones).to('cuda')

            batch = (O, G_d, G_a, A, R, O_n, G_d_n, G_a_n, dones)
            return batch
        else:
            O, G_d, G_a = [], [], []
            A = []
            R = []
            O_n, G_d_n, G_a_n = [], [], []
            dones = []
            for i in range(batch_size):
                sample = self.get_sample()
                O.append(sample.obs)
                G_d.append(sample.des_goal)
                G_a.append(sample.ach_goal)
                A.append(sample.action)
                R.append(sample.reward)
                O_n.append(sample.next_obs)
                G_d_n.append(sample.next_des_goal)
                G_a_n.append(sample.next_ach_goal)
                dones.append(sample.done)

            return torch.stack(O).to('cuda'), torch.stack(G_d).to('cuda'), torch.stack(G_a).to(
                'cuda'), torch.stack(A).to('cuda'), torch.stack(R).to('cuda'), torch.stack(O_n).to(
                'cuda'), torch.stack(G_d_n).to('cuda'), torch.stack(G_a_n).to('cuda'), torch.stack(dones).to(
                'cuda')

    def __len__(self):
        return len(self.data)
