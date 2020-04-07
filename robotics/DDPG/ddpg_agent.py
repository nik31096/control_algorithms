import torch

from collections import OrderedDict
from copy import deepcopy

from .utils import PolicyNetwork, ValueNetwork


class DDPG:
    def __init__(self,
                 observation_space_shape,
                 goal_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 writer,
                 mode='multi_env'):
        self.gamma = gamma
        self.tau = tau
        self.writer = writer
        self.mode = mode

        args = (observation_space_shape, goal_space_shape, action_space_shape, action_ranges)
        # online networks
        self.online_policy_network_gpu = PolicyNetwork(*args).to('cuda')
        self.online_policy_network_cpu = PolicyNetwork(*args)
        # syncronize gpu and cpu versions of online policy network
        self.online_policy_network_cpu.load_state_dict(self.online_policy_network_gpu.state_dict())
        self.online_policy_network_gpu.to('cuda')
        self.online_value_network = ValueNetwork(*args[:-1]).to('cuda')

        # target networks
        self.target_policy_network = PolicyNetwork(*args).to('cuda')
        self.target_value_network = ValueNetwork(*args[:-1]).to('cuda')

        # policy networks update
        self._hard_update(self.target_policy_network, self.online_policy_network_gpu)
        # value networks update
        self._hard_update(self.target_value_network, self.online_value_network)

        self.policy_opt = torch.optim.Adam(self.online_policy_network_gpu.parameters(), lr=1e-4)
        self.value_opt = torch.optim.Adam(self.online_value_network.parameters(), lr=1e-3, weight_decay=1e-2)

    def select_action(self, states, desired_goals, achieved_goals):
        actions = self.online_policy_network_cpu(states, desired_goals, achieved_goals)

        return actions

    def train(self, batch, epoch):
        obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = batch
        q_values = self.online_value_network(obs, des_goals, ach_goals, actions)
        q_n_values = self.target_value_network(next_obs, next_des_goals, next_ach_goals,
                                        self.target_policy_network(next_obs, next_des_goals, next_ach_goals).detach())

        y = rewards + self.gamma * (1 - dones) * q_n_values

        self.value_opt.zero_grad()
        value_loss = torch.mean((q_values - y) ** 2)
        value_loss.backward(retain_graph=True)
        self.value_opt.step()

        policy_loss = -torch.mean(self.online_value_network(obs, des_goals, ach_goals,
                                                       self.online_policy_network_gpu(obs, des_goals, ach_goals)))

        policy_loss.backward()
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        # logging loss values
        self.writer.add_scalar("Value loss", value_loss.cpu().data.numpy(), epoch)
        self.writer.add_scalar("Policy loss", -policy_loss.cpu().data.numpy(), epoch)

        # policy networks update
        self._soft_update(self.target_policy_network, self.online_policy_network_gpu, self.tau)
        # value networks update
        self._soft_update(self.target_value_network, self.online_value_network, self.tau)

    def syncronize_online_networks(self):
        state_dict = deepcopy(self.online_policy_network_gpu.state_dict())
        self.online_policy_network_cpu.load_state_dict(OrderedDict({key: state_dict[key].cpu() for key in state_dict}))

    def _hard_update(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, online, tau):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_((1 - tau)*target_param + tau*param)

    def save_models(self, path, model_name):
        torch.save(self.online_policy_network_gpu.state_dict(), "./weigths/policy_ddpg_2.pt")
        torch.save(self.online_value_network.state_dict(), "./weights/value_ddpg_2.pt")

    def load_pretrained(self, path, model_name):
        self.online_policy_network_gpu.load_state_dict(torch.load(f'{path}/policy_{model_name}.pt'))
        self.online_policy_network_cpu.load_state_dict(
            torch.load(f'{path}/policy_{model_name}.pt', map_location=torch.device('cpu')))
        self.online_value_network.load_state_dict(torch.load(f'{path}/value_{model_name}.pt'))
