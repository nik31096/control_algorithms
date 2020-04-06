import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import gym

from tqdm import trange
import numpy as np
from collections import namedtuple
import random


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, goal_shape, output_shape, action_ranges):
        super(PolicyNetwork, self).__init__()
        self.action_ranges = action_ranges
        self.layer_obs = nn.Linear(obs_shape, 200)
        self.layer_goal = nn.Linear(2*goal_shape, 200)
        self.layer1 = nn.Linear(400, 100)
        self.layer2 = nn.Linear(100, output_shape)
        self.layer3 = nn.Linear(100, output_shape)

    def forward(self, observation, desired_goal, achieved_goal):
        # observation = torch.FloatTensor(state['observation'])
        # goal = torch.cat([torch.FloatTensor(state['achieved_goal']),
        #                  torch.FloatTensor(state['desired_goal'])])
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
        sigma = torch.relu(self.layer3(out))

        distribution = torch.distributions.Normal(mu, sigma)

        action = distribution.sample() + torch.randn_like(mu)
        action = torch.clamp(action, self.action_ranges[0], self.action_ranges[1])

        return action


class ValueNetwork(nn.Module):
    def __init__(self, obs_shape, goal_shape, action_shape):
        super(ValueNetwork, self).__init__()
        self.layer_obs = nn.Linear(obs_shape, 200)
        self.layer_goal = nn.Linear(2*goal_shape, 200)
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





gamma = 0.95
EPS = 0.01


replay_buffer = ExperienceReplay()
online_policy_network = PolicyNetwork(env.observation_space["observation"].shape[0],
                                      env.observation_space["achieved_goal"].shape[0],
                                      env.action_space.shape[0],
                                      action_ranges=(env.action_space.low[0], env.action_space.high[0])).to('cuda')
online_value_network = ValueNetwork(env.observation_space["observation"].shape[0],
                                    env.observation_space["achieved_goal"].shape[0],
                                    env.action_space.shape[0]).to('cuda')

target_policy_network = PolicyNetwork(env.observation_space["observation"].shape[0],
                                      env.observation_space["achieved_goal"].shape[0],
                                      env.action_space.shape[0],
                                      action_ranges=(env.action_space.low[0], env.action_space.high[0])).to('cuda')
target_value_network = ValueNetwork(env.observation_space["observation"].shape[0],
                                    env.observation_space["achieved_goal"].shape[0],
                                    env.action_space.shape[0]).to('cuda')

pretrained = True
if pretrained:
    online_policy_network.load_state_dict(torch.load('./weights/policy_ddpg_1.pt'))
    online_value_network.load_state_dict(torch.load('./weights/value_ddpg_1.pt'))


def synchronize_policy_networks_params(eps):
    for policy_tensor in online_policy_network.state_dict():
        temp = (1 - eps) * target_policy_network.state_dict()[policy_tensor] +                                                     eps * online_policy_network.state_dict()[policy_tensor]
        target_policy_network.state_dict()[policy_tensor] = temp


def synchronize_value_networks_params(eps):
    for policy_tensor in online_value_network.state_dict():
        temp = (1 - eps) * target_value_network.state_dict()[policy_tensor] +                                                          eps * online_value_network.state_dict()[policy_tensor]
        target_value_network.state_dict()[policy_tensor] = temp


synchronize_policy_networks_params(eps=1)
synchronize_value_networks_params(eps=1)

policy_opt = torch.optim.Adam(online_policy_network.parameters(), lr=1e-3)
value_opt = torch.optim.Adam(online_value_network.parameters(), lr=1e-3)

# Putting some data in replay buffer
online_policy_network.cpu()
for _ in range(100):
    for step in range(1000):
        action = online_policy_network(state['observation'], state['desired_goal'], state['achieved_goal'])
        next_state, reward, done, _ = env.step(action.data.numpy())
        replay_buffer.put(state, action, reward, next_state, torch.FloatTensor([done]))
        if done:
            state = env.reset()
            break
        state = next_state

online_policy_network.to('cuda')

assert len(replay_buffer) != 0, "Replay buffer is empty!"

writer = SummaryWriter("./runs/run_1")

state = env.reset()
for epoch in trange(100000):
    for step in range(1000):
        action = online_policy_network.cpu()(state['observation'], state['desired_goal'], state['achieved_goal'])
        next_state, reward, done, _ = env.step(action.data.numpy())
        replay_buffer.put(state, torch.FloatTensor(action), torch.FloatTensor([reward]), next_state, torch.FloatTensor([done]))
        # TODO: add success rate instead of episode reward
        if done:
            writer.add_scalar("Success rate", 1 if reward == -0.0 else 0, epoch)
            state = env.reset()
            break
            
        state = next_state

        # Training
        online_policy_network.to('cuda')
        obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = replay_buffer.sample(3000)
        q_values = online_value_network(obs, des_goals, ach_goals, actions)
        q_n_values = target_value_network(next_obs, next_des_goals, next_ach_goals, 
                                          target_policy_network(next_obs, next_des_goals, next_ach_goals).detach())

        y = rewards + gamma*(1 - dones)*q_n_values

        value_loss = torch.mean((q_values - y) ** 2)
        writer.add_scalar("Value loss", value_loss.cpu().data.numpy(), epoch)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        policy_loss = -torch.mean(online_value_network(obs, des_goals, ach_goals, 
                                                       online_policy_network(obs, des_goals, ach_goals)))
        
        writer.add_scalar("Policy loss", torch.mean(online_value_network(obs, des_goals, ach_goals, 
                                                       online_policy_network(obs, des_goals, ach_goals))).cpu().data.numpy(), epoch)

        policy_loss.backward()
        policy_opt.step()
        policy_opt.zero_grad()
        
        synchronize_value_networks_params(eps=EPS)
        synchronize_policy_networks_params(eps=EPS)

torch.save(online_policy_network.state_dict(), "policy_ddpg_1.pt")
torch.save(online_value_network.state_dict(), "value_ddpg_1.pt")
