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

dynamics = NonLinearDynamics(state_dim=envs.observation_space['observation'].shape[0],
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
        loss = dynamics.train_dynamics(epochs=100, batch_size=512)
        losses.extend(loss)

plt.plot(losses)
plt.show()
dynamics.save('dynamics.pt')

print("[INFO] Dynamics testing")
env = gym.make(env_id)
real_state = env.reset()
pred_state = real_state
real_states, pred_states = [], []
while True:
    action = np.random.uniform(-1, 1, size=2)
    next_real_state, _, done, _ = env.step(action)
    next_pred_state = dynamics.dyn(pred_state, action)
    pred_states.append(next_pred_state)
    real_states.append(next_real_state['observation'])
    if done:
        break

    real_state = next_real_state
    pred_state = next_pred_state


plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot([x[0] for x in real_states], label='real')
plt.plot([x[0] for x in pred_states], label='predicted')
plt.title("First joint angle")
plt.legend()
plt.subplot(222)
plt.plot([x[1] for x in real_states], label='real')
plt.plot([x[1] for x in pred_states], label='predicted')
plt.title("First joint angular velocity")
plt.legend()
plt.subplot(223)
plt.plot([x[2] for x in real_states], label='real')
plt.plot([x[2] for x in pred_states], label='predicted')
plt.title("Second joint angle")
plt.legend()
plt.subplot(224)
plt.plot([x[3] for x in real_states], label='real')
plt.plot([x[3] for x in pred_states], label='predicted')
plt.title("Second joint angular velocity")
plt.legend()
plt.show()
