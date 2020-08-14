import gym
import Reach_v0

from robotics.OptimalControl.DDP import DDP
from tqdm import trange

import numpy as np
import autograd.numpy as anp
from tensorboardX import SummaryWriter

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


env = gym.make('Reach-v2')
T = env._max_episode_steps
print("Horizon", T)

run_name = "ddp_1"
writer = SummaryWriter(f'runs/{run_name}')

ddp = DDP(a1=1,
          a2=1,
          a3=1,
          dynamics_model=env.get_next_state,
          forward_dynamics=env.forward_dynamics,
          umax=1,
          state_dim=env.observation_space["observation"].shape[0],
          pred_time=T
          )


def execute_actions(initial_state, actions_sequence):
    states_sequence = [initial_state]
    state = initial_state
    for t in range(ddp.pred_time):
        next_state = env.get_next_state(state, actions_sequence[t])
        states_sequence.append(next_state)
        state = next_state

    return states_sequence


state = env.reset()
goal = anp.array([1, 1, state['desired_goal'][2]], dtype=float)
env.set_goal(goal)
np.save(f"./weights/{run_name}_goal", goal)
ddp.set_goal(goal[:2])
global_actions = []

actions = [1e-2*np.random.random(size=2) for _ in range(ddp.pred_time)]
states = execute_actions(state['observation'], actions)

for epoch in trange(20):
    positive = False
    while not positive:
        k, K, positive, delta_J = ddp.backward(states, actions)

    restart = False
    while not restart:
        states, actions = ddp.forward(states, actions, k, K, delta_J)

    theta1, theta2 = states[-1][0], states[-1][2]
    achieved_goal = ddp.forward_dynamics(theta1, theta2)
    distance = np.linalg.norm(achieved_goal - goal[:2])
    writer.add_scalar("Final_distance", distance, epoch)

    np.save(f"weights/{run_name}_actions", np.array(actions))

    state = env.reset()
    env.set_goal(goal)
    distances = []
    for t in range(T):
        next_state, reward, done, _ = env.step(actions[t])
        env.render()
        if done:
            break
