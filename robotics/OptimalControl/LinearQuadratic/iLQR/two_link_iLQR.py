import numpy as np
import matplotlib.pyplot as plt

import gym
import Reach_v0

from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR

from tqdm import trange

env = gym.make('Reach-v3')
state = env.reset()

epochs = 20

dyn = env.get_next_state
final_state = env.inverse_kinematics(x=state['desired_goal'][0], y=state['desired_goal'][1])
print("Final state", final_state)
final_state = np.array([final_state[0], 0, final_state[1], 0])
ilqr = iLQR(dynamics=dyn, final_state=final_state, state_dim=4, action_dim=2, horizon=200)
optimal_controls = ilqr.fit_controller(epochs=20, verbose=1)

t1, t2, controls = [state['observation'][0]], [state['observation'][2]], []

for t in trange(2000):
    control = ilqr.get_control(state['observation'], t % 30)
    next_state, reward, done, _ = env.step(control)
    t1.append(next_state['observation'][0])
    t2.append(next_state['observation'][2])
    controls.append(control)
    state = next_state

    if t % 30 == 0:
        optimal_controls = ilqr.fit_controller(controls=optimal_controls, epochs=8,
                                               initial_state=state['observation'], verbose=0)

ilqr.save_controller('reach_known_dyn_mpc')

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.plot([0, 2000], [final_state[0], final_state[0]], label="Desired state")
plt.plot(t1, label="Achieved state")
plt.title("Theta 1")
plt.legend()
plt.subplot(222)
plt.plot([0, 2000], [final_state[2], final_state[2]], label="Desired state")
plt.plot(t2, label="Achieved")
plt.title("Theta 2")
plt.legend()
plt.subplot(223)
plt.plot([a[0] for a in controls])
plt.title("First joint torque")
plt.subplot(224)
plt.plot([a[1] for a in controls])
plt.title("Second joint torque")
plt.show()

state = env.reset()
for t in range(2000):
    next_state, reward, done, _ = env.step(controls[t])
    env.render()
