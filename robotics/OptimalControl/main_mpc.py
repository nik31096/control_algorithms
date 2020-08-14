import gym
import Reach_v0

from robotics.OptimalControl.DDP import DDP_v1

import numpy as np

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)

env = gym.make('Reach-v2')
T = env._max_episode_steps
print("Horizon", T)

run_name = "ddp_2"

ddp = DDP_v1(a1=100,
             a2=10,
             a3=10,
             dynamics_model=env.get_next_state,
             forward_dynamics=env.forward_dynamics,
             umax=1,
             state_dim=env.observation_space["observation"].shape[0],
             pred_time=200
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
goal = state['desired_goal']
initial_distance = np.linalg.norm(state["achieved_goal"] - state['desired_goal'])

np.save(f"{run_name}_goal", goal)
ddp.set_goal(goal[:2])
global_actions = []

actions = [1e-2*np.random.random(size=2) for _ in range(ddp.pred_time)]
states = execute_actions(state['observation'], actions)

H = 20
assert H <= ddp.pred_time
iteration = 0
while True:
    restart_backward = True
    while restart_backward:
        k, K, restart_backward = ddp.backward(states, actions, iteration)

    restart_forward = True
    while restart_forward:
        new_states, new_actions, restart_forward = ddp.forward(states, actions, k, K, iteration)

    for inner_time in range(H):
        next_state, reward, done, _ = env.step(new_actions[inner_time])

    distance = np.linalg.norm(next_state['achieved_goal'] - goal)
    print(f"Distance to the goal: {distance}, Initial distance: {initial_distance}")
    global_actions.extend(new_actions[:H])

    if distance < env.distance_threshold or iteration*H >= T - 1:
        break

    actions = [a for a in new_actions[H:]] + [new_actions[-1] for _ in range(H)]
    states = execute_actions(next_state['observation'], actions)

    iteration += 1

actions = np.array(global_actions + [np.zeros(2) for _ in range(T - len(global_actions))])
np.save(f"{run_name}_actions", actions)

# testing
state = env.reset()
env.set_goal(goal)
for t in range(T):
    next_state, reward, done, _ = env.step(global_actions[t])
    env.render()
    if done:
        break
