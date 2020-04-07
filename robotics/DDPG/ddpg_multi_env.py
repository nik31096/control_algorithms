import torch
from tensorboardX import SummaryWriter
import gym

from tqdm import trange
from copy import deepcopy
from collections import OrderedDict
import numpy as np

from utils import ExperienceReplay, PolicyNetwork, ValueNetwork
from multiprocessing_environment.subproc_env import SubprocVecEnv


def make_env(env_id):
    def _f():
        env = gym.make(env_id)
        return env
    return _f


env_id = "FetchReach-v1"
n_envs = 8

envs = [make_env(env_id) for _ in range(n_envs)]
envs = SubprocVecEnv(envs, context='fork', in_series=1)
states = envs.reset()


gamma = 0.95
EPS = 0.01
batch_size = 512
writer_name = "./runs/run_3"

replay_buffer = ExperienceReplay()

online_policy_network_gpu = PolicyNetwork(envs.observation_space["observation"].shape[0],
                                          envs.observation_space["achieved_goal"].shape[0],
                                          envs.action_space.shape[0],
                                          action_ranges=(envs.action_space.low[0], envs.action_space.high[0]))
online_policy_network_cpu = PolicyNetwork(envs.observation_space["observation"].shape[0],
                                          envs.observation_space["achieved_goal"].shape[0],
                                          envs.action_space.shape[0],
                                          action_ranges=(envs.action_space.low[0], envs.action_space.high[0]))
online_policy_network_cpu.load_state_dict(online_policy_network_gpu.state_dict())
online_policy_network_gpu.to('cuda')

online_value_network = ValueNetwork(envs.observation_space["observation"].shape[0],
                                    envs.observation_space["achieved_goal"].shape[0],
                                    envs.action_space.shape[0]).to('cuda')

target_policy_network = PolicyNetwork(envs.observation_space["observation"].shape[0],
                                      envs.observation_space["achieved_goal"].shape[0],
                                      envs.action_space.shape[0],
                                      action_ranges=(envs.action_space.low[0], envs.action_space.high[0])).to('cuda')
target_value_network = ValueNetwork(envs.observation_space["observation"].shape[0],
                                    envs.observation_space["achieved_goal"].shape[0],
                                    envs.action_space.shape[0]).to('cuda')

pretrained = False
if pretrained:
    online_policy_network_gpu.load_state_dict(torch.load('./weights/policy_ddpg_2.pt'))
    online_policy_network_cpu.load_state_dict(torch.load('./weights/policy_ddpg_2.pt', map_location=torch.device('cpu')))
    online_value_network.load_state_dict(torch.load('./weights/value_ddpg_2.pt'))


def synchronize_policy_networks_params(eps):
    for policy_tensor in online_policy_network_gpu.state_dict():
        temp = (1 - eps) * target_policy_network.state_dict()[policy_tensor] + \
               eps * online_policy_network_gpu.state_dict()[policy_tensor]
        target_policy_network.state_dict()[policy_tensor] = temp


def synchronize_value_networks_params(eps):
    for policy_tensor in online_value_network.state_dict():
        temp = (1 - eps) * target_value_network.state_dict()[policy_tensor] + \
               eps * online_value_network.state_dict()[policy_tensor]
        target_value_network.state_dict()[policy_tensor] = temp


synchronize_policy_networks_params(eps=1)
synchronize_value_networks_params(eps=1)

policy_opt = torch.optim.Adam(online_policy_network_gpu.parameters(), lr=1e-3)
value_opt = torch.optim.Adam(online_value_network.parameters(), lr=1e-3)

writer = SummaryWriter(writer_name)
distance_writers = [SummaryWriter(f'{writer_name}/distance_writer.env_{i}') for i in range(n_envs)]

for epoch in trange(1000):
    for step in range(1000):
        actions = online_policy_network_cpu(states['observation'], states['desired_goal'], states['achieved_goal'])
        next_states, rewards, dones, info = envs.step(actions.data.numpy())
        replay_buffer.put(states, actions, rewards, next_states, dones)
        for i in range(n_envs):
            if dones[i]:
                distance = np.linalg.norm(states['desired_goal'][i] - states['achieved_goal'][i])
                envs.reset_env(env_index=i)
                distance_writers[i].add_scalar("Desired-achieved distance", distance, epoch)
        states = next_states

        if len(replay_buffer) > batch_size:
            # Training
            # for i in range(10 if len(replay_buffer) < 100000 else 100):
            obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = \
                replay_buffer.sample(batch_size)
            q_values = online_value_network(obs, des_goals, ach_goals, actions)
            next_actions = target_policy_network(next_obs, next_des_goals, next_ach_goals)
            q_n_values = target_value_network(next_obs, next_des_goals, next_ach_goals, next_actions)

            y = rewards + gamma*(1 - dones)*q_n_values

            value_loss = torch.mean((q_values - y) ** 2)
            writer.add_scalar("Value loss", value_loss.cpu().data.numpy(), epoch)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            policy_loss = -torch.mean(online_value_network(obs, des_goals, ach_goals,
                                                           online_policy_network_gpu(obs, des_goals, ach_goals)))
            writer.add_scalar("Policy loss", -policy_loss.cpu().data.numpy(), epoch)

            policy_loss.backward()
            policy_opt.step()
            policy_opt.zero_grad()

            synchronize_value_networks_params(eps=EPS)
            synchronize_policy_networks_params(eps=EPS)

            # state_dict = deepcopy(online_policy_network_gpu.state_dict())
            # online_policy_network_cpu.load_state_dict(OrderedDict({key: state_dict[key].cpu() for key in state_dict}))

    if (epoch + 1) % 50 == 0:
        torch.save(online_policy_network_gpu.state_dict(), "./weights/policy_ddpg_2.pt")
        torch.save(online_value_network.state_dict(), "./weights/value_ddpg_2.pt")

replay_buffer.save('./buffer')