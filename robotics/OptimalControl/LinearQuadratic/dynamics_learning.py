import gym
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing_environment.subproc_env import SubprocVecEnv
from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import NonLinearDynamics

from tqdm import trange

import Reach_v0


def make_env(env_id):
    def _f():
        env = gym.make(env_id)
        return env

    return _f


env_id = "Reach-v3"
n_envs = 32

envs = [make_env(env_id) for _ in range(n_envs)]
envs = SubprocVecEnv(envs, context='fork', in_series=8)
states = envs.reset()

dynamics = Dynamics(state_dim=envs.observation_space['observation'].shape[0],
                    action_dim=envs.action_space.shape[0],
                    device='cuda:0')

losses = []
for epoch in trange(100):
    # data collection stage
    episode_states = []
    episode_actions = []
    episode_next_states = []
    while True:
        actions = [envs.action_space.sample() for _ in range(n_envs)]
        next_states, _, dones, _ = envs.step(actions)
        episode_states.append(states['observation'])
        episode_actions.extend(actions)
        episode_next_states.append(next_states['observation'])
        states = next_states
        if np.all(dones):
            states = envs.reset()
            episode_states = np.concatenate(episode_states, axis=0)
            episode_actions = np.array(episode_actions)
            episode_next_states = np.concatenate(episode_next_states, axis=0)
            dynamics.add_trajectory(states=episode_states, actions=episode_actions, next_states=episode_next_states)
            break

    if epoch % 5 == 0:
        loss = dynamics.train_dynamics()
        losses.extend(loss)

plt.plot(losses)
plt.show()

dynamics.save('dynamics.pt')
