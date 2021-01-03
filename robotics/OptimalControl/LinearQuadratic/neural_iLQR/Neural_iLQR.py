import gym
import Reach_v0

from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import NonLinearDynamics, LinearDynamics
from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR

import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt


env = gym.make('Reach-v3')
max_steps = env._max_episode_steps
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]

use_pretrained_dyn = False
dynamics = NonLinearDynamics(state_dim=state_dim, action_dim=action_dim, device='cuda')

ilqr = iLQR(dynamics=dynamics, initial_state=np.zeros((state_dim, )), horizon=200)

state = env.reset()
final_state = state['desired_goal']
ilqr.set_final_state(final_state=final_state, inv_kin=env.inverse_kinematics)

optimal_controls = ilqr.fit_controller(epochs=1000, verbose=1)

final_states = []
epochs = 1000
for epoch in trange(epochs):
    controls, states, next_states = [], [], []
    # online data collection
    for t in range(max_steps):
        u = ilqr.get_control(state['observation'], t % 10)
        next_state, reward, done, _ = env.step(u)
        states.append(state['observation'])
        controls.append(u)
        next_states.append(next_state['observation'])

        if t % 10 == 0:
            ilqr.fit_controller(controls=controls, epochs=5)

        state = next_state

    final_states.append(state['observation'])

    dynamics.add_trajectory(states=states, actions=controls, next_states=next_states)
    if epoch % 5 == 0 and epoch > 50:
        dynamics.train_dynamics()

    state = env.reset()

plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot([x[0] for x in final_states])
plt.plot([0, epochs], [final_state[0], final_state[0]])
plt.title("First joint final angle")
plt.subplot(222)
plt.plot([x[2] for x in final_states])
plt.plot([0, epochs], [final_state[1], final_state[1]])
plt.title("Second joint final angle")
plt.subplot(223)
plt.plot([x[1] for x in final_states])
plt.title("First joint final speed")
plt.subplot(224)
plt.plot([x[3] for x in final_states])
plt.title("Second joint final speed")
plt.show()

dynamics.save('dynamics_reach_1000.pt')
ilqr.save_controller('reach_mpc_1', controls)

state = env.reset()
states = [state['observation']]
actions = []
for t in range(max_steps):
    u = ilqr.get_control(state['observation'], t)
    next_state, reward, done, _ = env.step(u)
    state = next_state
    states.append(state['observation'])
    actions.append(u)
    # env.render()


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
