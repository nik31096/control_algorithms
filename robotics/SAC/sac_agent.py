import torch

from robotics.SAC.utils import QNetwork, ValueNetwork, PolicyNetwork, Normalizer

import os


class SACAgent_v0:
    def __init__(self,
                 observation_space_shape,
                 goal_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 q_lr,
                 value_lr,
                 policy_lr,
                 train_device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = train_device

        self.alpha = torch.ones(1, requires_grad=True)

        self.value_network = ValueNetwork(observation_space_shape, goal_space_shape).to(self.device)
        self.target_value_network = ValueNetwork(observation_space_shape, goal_space_shape).to(self.device)
        self.q_network_1 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)
        self.q_network_2 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)
        self.policy_network = PolicyNetwork(observation_space_shape, goal_space_shape,
                                            action_space_shape, action_ranges).to(self.device)
        self.online_policy_network = PolicyNetwork(observation_space_shape, goal_space_shape,
                                                   action_space_shape, action_ranges)
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())

        self.q_1_opt = torch.optim.Adam(self.q_network_1.parameters(), lr=q_lr)
        self.q_2_opt = torch.optim.Adam(self.q_network_2.parameters(), lr=q_lr)
        self.v_opt = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

    def select_action(self, states):
        observations = torch.FloatTensor(states['observation'])
        desired_goals = torch.FloatTensor(states['desired_goal'])
        achieved_goals = torch.FloatTensor(states['achieved_goal'])
        goals = torch.cat([desired_goals, achieved_goals], dim=-1)
        actions = self.online_policy_network(observations, goals, mode='eval')

        return actions.data.numpy()

    def value_network_update(self, obs, des_goals, ach_goals):
        goals = torch.cat([des_goals, ach_goals], dim=-1)

        v_s = self.value_network(obs, goals)
        actions, log_probs = self.policy_network(obs, goals)
        q_1 = self.q_network_1(obs, goals, actions).detach()
        q_2 = self.q_network_2(obs, goals, actions).detach()
        q = torch.min(q_1, q_2)
        loss = torch.mean(0.5*(v_s - (q - self.alpha*log_probs.detach()))**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.1)
        self.v_opt.step()
        self.v_opt.zero_grad()

        if 'cuda' in self.device:
            return loss.cpu().data.numpy()
        else:
            return loss.data.numpy()

    def q_network_update(self, obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones):
        self.q_1_opt.zero_grad()
        self.q_2_opt.zero_grad()

        goals = torch.cat([des_goals, ach_goals], dim=-1)
        next_goals = torch.cat([next_des_goals, next_ach_goals], dim=-1)
        q_hat = rewards + (1 - dones)*self.gamma*self.target_value_network(next_obs, next_goals).detach()

        loss_1 = torch.mean(0.5*(self.q_network_1(obs, goals, actions) - q_hat)**2)
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=0.1)
        self.q_1_opt.step()
        loss_2 = torch.mean(0.5*(self.q_network_2(obs, goals, actions) - q_hat)**2)
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_2.parameters(), max_norm=0.1)
        self.q_2_opt.step()

        if 'cuda' in self.device:
            return loss_1.cpu().data.numpy(), loss_2.cpu().data.numpy()
        else:
            return loss_1.data.numpy(), loss_2.data.numpy()

    def policy_network_update(self, obs, des_goals, ach_goals):
        goals = torch.cat([des_goals, ach_goals], dim=-1)
        actions, log_probs = self.policy_network(obs, goals)
        q_1 = self.q_network_1(obs, goals, actions)
        q_2 = self.q_network_2(obs, goals, actions)
        q = torch.min(q_1, q_2)
        loss = torch.mean(self.alpha*log_probs - q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.1)
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        if 'cuda' in self.device:
            return loss.cpu().data.numpy()
        else:
            return loss.data.numpy()

    def train(self, batch):
        obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = batch
        value_loss = self.value_network_update(obs, des_goals, ach_goals)
        q_1_loss, q_2_loss = self.q_network_update(obs, des_goals, ach_goals, actions, rewards, next_obs,
                                                   next_des_goals, next_ach_goals, dones)
        policy_loss = self.policy_network_update(obs, des_goals, ach_goals)

        self._soft_update(self.tau)
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())
        assert self.online_policy_network.layer2.weight.device == torch.device('cpu'), \
            "Online network was placed on cuda or somewhere else. Please check the location!"

        return value_loss, q_1_loss, q_2_loss, policy_loss

    def _soft_update(self, tau):
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_((1 - tau)*target_param + tau*param)

    def save_models(self, model_name):
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}_policy_network.pt')
        torch.save(self.q_network_1.state_dict(), f'./weights/{model_name}_q_1_network.pt')
        torch.save(self.q_network_2.state_dict(), f'./weights/{model_name}_q_2_network.pt')
        torch.save(self.value_network.state_dict(), f'./weights/{model_name}_value_network.pt')
        torch.save(self.target_value_network.state_dict(), f'./weights/{model_name}_target_value_network.pt')

    def load_pretrained_models(self, model_name):
        if not os.path.exists('./weights'):
            print("[INFO] No ./weights folder to load pretrained models")
            raise Exception

        self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}_policy_network.pt"))
        self.q_network_1.load_state_dict(torch.load(f"./weights/{model_name}_q_1_network.pt"))
        self.q_network_2.load_state_dict(torch.load(f"./weights/{model_name}_q_2_network.pt"))
        self.value_network.load_state_dict(torch.load(f"./weights/{model_name}_value_network.pt"))
        self.target_value_network.load_state_dict(torch.load(f"./weights/{model_name}_target_value_network.pt"))


class SACAgent_v1:
    def __init__(self,
                 observation_space_shape,
                 goal_space_shape,
                 action_space_shape,
                 action_ranges,
                 gamma,
                 tau,
                 q_lr,
                 alpha_lr,
                 policy_lr,
                 train_device='cuda',
                 mode='multi_env'
                 ):
        self.gamma = gamma
        self.tau = tau
        self.mode = mode
        self.device = train_device

        self.obs_norm = Normalizer(observation_space_shape)
        self.goal_norm = Normalizer(goal_space_shape)

        self.q_network_1 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)
        self.q_network_2 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)

        self.target_q_network_1 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)
        self.target_q_network_2 = QNetwork(observation_space_shape, goal_space_shape, action_space_shape).to(self.device)
        self.target_q_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_q_network_2.load_state_dict(self.q_network_2.state_dict())

        self.policy_network = PolicyNetwork(observation_space_shape, goal_space_shape,
                                            action_space_shape, action_ranges).to(self.device)
        self.online_policy_network = PolicyNetwork(observation_space_shape, goal_space_shape,
                                                   action_space_shape, action_ranges)
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())

        self.q_1_opt = torch.optim.Adam(self.q_network_1.parameters(), lr=q_lr)
        self.q_2_opt = torch.optim.Adam(self.q_network_2.parameters(), lr=q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # alpha part
        self.alpha = 0.2
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -torch.FloatTensor([action_space_shape]).to(self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, states, mode='train'):
        observations = states['observation']
        desired_goals = states['desired_goal']
        achieved_goals = states['achieved_goal']
        goals = desired_goals - achieved_goals

        if mode == 'train':
            self.obs_norm.update_stats(observations)
            self.goal_norm.update_stats(goals)

        observations = self.obs_norm.normalize(observations)
        goals = self.goal_norm.normalize(goals)

        observations = torch.FloatTensor(observations)
        goals = torch.FloatTensor(goals)

        if mode == 'train':
            actions, _ = self.online_policy_network(observations, goals, mode=mode)
        else:
            actions = self.online_policy_network(observations, goals, mode=mode)

        return actions.data.numpy()

    def q_network_update(self, obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones):
        self.q_1_opt.zero_grad()
        self.q_2_opt.zero_grad()

        goals = des_goals - ach_goals
        next_goals = next_des_goals - next_ach_goals

        next_actions, next_log_probs = self.policy_network(next_obs, next_goals)
        target_q_1_next = self.target_q_network_1(next_obs, next_goals, next_actions)
        target_q_2_next = self.target_q_network_2(next_obs, next_goals, next_actions)
        min_q_target = torch.min(target_q_1_next, target_q_2_next)
        next_value_function = min_q_target - self.alpha * next_log_probs
        q_hat = rewards + (1 - dones)*self.gamma*next_value_function.detach()

        loss_1 = torch.mean(0.5*(self.q_network_1(obs, goals, actions) - q_hat)**2)
        loss_1.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=0.1)
        self.q_1_opt.step()
        loss_2 = torch.mean(0.5*(self.q_network_2(obs, goals, actions) - q_hat)**2)
        loss_2.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network_2.parameters(), max_norm=0.1)
        self.q_2_opt.step()

        if 'cuda' in self.device:
            return loss_1.cpu().data.numpy(), loss_2.cpu().data.numpy()
        else:
            return loss_1.data.numpy(), loss_2.data.numpy()

    def policy_network_update(self, obs, des_goals, ach_goals):
        goals = des_goals - ach_goals
        actions, log_probs = self.policy_network(obs, goals)
        q_1 = self.q_network_1(obs, goals, actions)
        q_2 = self.q_network_2(obs, goals, actions)
        q = torch.min(q_1, q_2)
        loss = torch.mean(self.alpha*log_probs - q)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.1)
        self.policy_opt.step()
        self.policy_opt.zero_grad()

        if 'cuda' in self.device:
            return loss.cpu().data.numpy()
        else:
            return loss.data.numpy()

    def alpha_update(self, obs, des_goals, ach_goals):
        goals = des_goals - ach_goals
        actions, log_probs = self.policy_network(obs, goals)

        loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        loss.backward()
        self.alpha_optim.step()
        self.alpha_optim.zero_grad()

        self.alpha = self.log_alpha.exp()

        if 'cuda' in self.device:
            return loss.cpu().data.numpy(), self.alpha.item()
        else:
            return loss.data.numpy(), self.alpha.item()

    def train(self, batch):
        obs, des_goals, ach_goals, actions, rewards, next_obs, next_des_goals, next_ach_goals, dones = batch
        q_1_loss, q_2_loss = self.q_network_update(obs, des_goals, ach_goals, actions, rewards, next_obs,
                                                   next_des_goals, next_ach_goals, dones)
        policy_loss = self.policy_network_update(obs, des_goals, ach_goals)
        entropy_loss, alpha = self.alpha_update(obs, des_goals, ach_goals)

        self._soft_update(self.target_q_network_1, self.q_network_1, self.tau)
        self._soft_update(self.target_q_network_2, self.q_network_2, self.tau)

        # check online cpu policy network location and copy trained weights into it to play in the environment
        self.online_policy_network.load_state_dict(self.policy_network.state_dict())
        assert self.online_policy_network.layer2.weight.device == torch.device('cpu'), \
            "Online network was placed on cuda or somewhere else. Please check the location!"

        return q_1_loss, q_2_loss, policy_loss, entropy_loss, alpha

    @staticmethod
    def _soft_update(online, target, tau):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_((1 - tau)*target_param.data + tau*param.data)

    def save_models(self, model_name):
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        if not os.path.exists(f'./weights/{model_name}'):
            os.mkdir(f'./weights/{model_name}')

        torch.save(self.policy_network.state_dict(), f'./weights/{model_name}/{model_name}_policy_network.pt')
        torch.save(self.q_network_1.state_dict(), f'./weights/{model_name}/{model_name}_q_1_network.pt')
        torch.save(self.q_network_2.state_dict(), f'./weights/{model_name}/{model_name}_q_2_network.pt')
        torch.save(self.target_q_network_1.state_dict(), f'./weights/{model_name}/{model_name}_target_q_network_1.pt')
        torch.save(self.target_q_network_2.state_dict(), f'./weights/{model_name}/{model_name}_target_q_network_2.pt')

    def load_pretrained_models(self, model_name):
        if not os.path.exists('./weights'):
            raise IOError("No ./weights folder to load pretrained models")
        if not os.path.exists(f'./weights/{model_name}'):
            raise IOError(f"No ./weights/{model_name} folder to load pretrained model")

        self.policy_network.load_state_dict(torch.load(f"./weights/{model_name}/{model_name}_policy_network.pt"))
        self.q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/{model_name}_q_1_network.pt"))
        self.q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/{model_name}_q_2_network.pt"))
        self.target_q_network_1.load_state_dict(torch.load(f"./weights/{model_name}/{model_name}_target_q_network_1.pt"))
        self.target_q_network_2.load_state_dict(torch.load(f"./weights/{model_name}/{model_name}_target_q_network_2.pt"))
