import gym
import numpy as np
from multiprocessing_environment.subproc_env import SubprocVecEnv
from robotics.DDPG.utils import ExperienceReplay

import time
from tqdm import trange

start = time.time()


def make_env(env_id):
    def _f():
        env = gym.make(env_id)
        return env
    return _f


env_id = "FetchReach-v1"
n_envs = 8

envs = [make_env(env_id) for _ in range(n_envs)]
envs = SubprocVecEnv(envs, context='fork')
states = envs.reset()

collect = True
if collect:
    for i in range(10):
        replay = ExperienceReplay(size=5000000, mode='multi_env')
        for _ in trange(round(replay.size / n_envs)):
            actions = np.array([envs.action_space.sample() for _ in range(n_envs)])
            next_states, rewards, dones, _ = envs.step(actions)
            replay.put(states, actions, rewards, next_states, dones)

        replay.save(f'./exp_replay_{i}')
else:
    replay = ExperienceReplay(size=5000000, mode='multi_env')
    data = replay.load('./exp_replay_1')
