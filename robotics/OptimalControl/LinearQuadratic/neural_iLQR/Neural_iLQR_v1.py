import gym
import Reach_v0
from multiprocessing_environment.subproc_env import SubprocVecEnv

from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import NonLinearDynamics, LinearDynamics
from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR

import numpy as np
from matplotlib import pyplot as plt

from tqdm import trange


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

env = gym.make(env_id)
max_steps = env._max_episode_steps
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]

dynamics = NonLinearDynamics(state_dim=envs.observation_space['observation'].shape[0],
                             action_dim=envs.action_space.shape[0],
                             device='cuda:0')

print("[INFO] Model learning")
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
        dynamics.train_dynamics(epochs=100, batch_size=512)


ilqr = iLQR(dynamics=dynamics, initial_state=np.zeros((state_dim, )), horizon=100)

state = env.reset()
final_state = state['desired_goal']
ilqr.set_final_state(final_state=final_state, inv_kin=env.inverse_kinematics)

print("[INFO] Initial iLQR fitting")
optimal_controls = ilqr.fit_controller(epochs=20, verbose=1)

print("[INFO] iLQR MPC starts")
final_states = []
mpc_timesteps = 1
controls, states, next_states = [], [], []
for t in range(max_steps):
    u = ilqr.get_control(state['observation'], t % mpc_timesteps)
    next_state, reward, done, _ = env.step(u)
    states.append(state['observation'])
    controls.append(u)
    next_states.append(next_state['observation'])

    if t % mpc_timesteps == 0 and t != 0:
        optimal_controls = ilqr.fit_controller(controls=optimal_controls, epochs=8,
                                               initial_state=next_state['observation'])

    state = next_state


state = env.reset()
states = [state['observation']]
actions = []
for t in range(max_steps):
    u = ilqr.get_control(state['observation'], t)
    next_state, reward, done, _ = env.step(u)
    state = next_state
    states.append(state['observation'])
    actions.append(u)
    env.render()


plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot([x[0] for x in states])
plt.plot([0, max_steps], [final_state[0], final_state[0]])
plt.title("First joint angle")
plt.subplot(222)
plt.plot([a[0] for a in actions])
plt.title("First joint torque")
plt.subplot(223)
plt.plot([x[2] for x in states])
plt.plot([0, max_steps], [final_state[1], final_state[1]])
plt.title("Second joint angle")
plt.subplot(224)
plt.plot([a[1] for a in actions])
plt.title("Second joint torque")
plt.show()
