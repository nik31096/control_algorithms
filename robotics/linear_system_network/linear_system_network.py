import torch
import gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange

from robotics.linear_system_network.linear_system_network import ExperienceReplay, Network
from robotics.linear_system_network.linear_system_network import SubprocVecEnv


def make_env(env_id):
    def _f():
        env = gym.make(env_id)
        return env
    return _f


env_id = "FetchReach-v1"
n_envs = 24

envs = [make_env(env_id) for _ in range(n_envs)]
envs = SubprocVecEnv(envs, context='fork', in_series=3)
states = envs.reset()

writer = SummaryWriter('./runs/run1')

replay = ExperienceReplay(size=1000000)

# Интересный вопрос связан с выбором состояния: если выбирать в качестве состояния только позицию гриппера в 3d,
# то есть действия, которые не изменят эту позицию (то есть действительное положение манипулятора изменилось,
# но гриппер остался на том же месте).
# Выучить такую динамику намного сложнее, чем если бы мы формулировали состояние в полном виде (в виде
# 13-мерного верктора). Но, с другой стороны, нужно формулировать желаемое состояние, и если бы наше состояние было
# бы 13-ти мерным, то нужно было бы фиксировать все углы манипулятора, что заметно усложняло бы постановку задачи.

n = envs.observation_space["achieved_goal"].shape[0]
o = envs.observation_space["observation"].shape[0]
m = envs.action_space.shape[0]

print(f"State dim is: {n}, action dim is: {m}")
state_network = Network(input_dim=n + o, output_shape=(n, n)).to('cuda')
action_network = Network(input_dim=m + o, output_shape=(n, m)).to('cuda')

opt = torch.optim.Adam(list(state_network.parameters()) + list(action_network.parameters()), lr=1e-3)

epoch2save = 5000
batch_size = 512

for epoch in trange(1000000):
    actions = np.array([envs.action_space.sample() for _ in range(n_envs)])
    next_states, rewards, dones, _ = envs.step(actions)
    replay.put(states, actions, next_states)

    if len(replay) > batch_size:
        for iteration in range(100):
            obs_pos, pos, obs_actions, actions, next_pos = replay.sample(batch_size)

            A = state_network(obs_pos)
            B = action_network(obs_actions)

            next_states_ = torch.bmm(A, pos[:, :, np.newaxis]) + torch.bmm(B, actions[:, :, np.newaxis])

            loss = torch.mean((next_pos - next_states_.squeeze()) ** 2)

            writer.add_scalar("Loss", loss, epoch)
            loss.backward()
            opt.step()
            opt.zero_grad()

    if (epoch + 1) % epoch2save == 0:
        torch.save("./weights/state_network.pt", state_network.state_dict())
        torch.save("./weights/action_network.pt", action_network.state_dict())

# TODO: get the way to save experience replay buffer on disk
