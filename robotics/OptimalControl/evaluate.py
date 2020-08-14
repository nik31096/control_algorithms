import gym
import Reach_v0

import numpy as np

env = gym.make("Reach-v2")
T = env._max_episode_steps

actions = np.load("./weights/ddp_1_actions.npy")
#goal = np.load('./weights/ddp_5_goal.npy')
#env.set_goal(goal)
state = env.reset()
assert actions.shape[0] == T
print(actions.shape)

for t in range(T):
    print(actions[t])
    next_state, reward, done, _ = env.step(actions[t])

    env.render()
    if done:
        break

