import gym
import Reach_v0

from autograd import jacobian
import numpy as np
import matplotlib.pyplot as plt

from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import Dynamics

env = gym.make('Reach-v3')
state = env.reset()['observation']

nn_dyn = Dynamics(state_dim=4, action_dim=2, device='cpu')
nn_dyn.load('./dynamics.pt')
dyn = env.get_next_state
arg = 0
j = jacobian(dyn, arg)

nn_acc = []
batch = []
forward_acc = []
backward_acc = []
center_acc = []

real_states = [state]
pred_states = [state]

for t in range(env._max_episode_steps):
    action = env.action_space.sample()
    auto_jacobian = j(state, action)

    nn_acc.append(np.mean((auto_jacobian - nn_dyn.derivative(arg=arg, method='nn')(state, action))**2))
    forward_acc.append(np.mean((auto_jacobian - nn_dyn.derivative(arg=arg, method='forward')(state, action))**2))
    backward_acc.append(np.mean((auto_jacobian - nn_dyn.derivative(arg=arg, method='backward')(state, action))**2))
    center_acc.append(np.mean((auto_jacobian - nn_dyn.derivative(arg=arg, method='center')(state, action))**2))

    next_state, reward, done, _ = env.step(action)
    next_pred_state = nn_dyn.dyn(pred_states[-1], action)
    real_states.append(next_state['observation'])
    pred_states.append(next_pred_state)
    state = next_state['observation']

plt.plot(nn_acc, label='PyTorch jacobian method', linewidth=0.4)
plt.plot(forward_acc, label='Forward FD', linewidth=0.4)
plt.plot(backward_acc, label='Backward FD', linewidth=0.4)
plt.plot(center_acc, label='Centered FD', linewidth=0.4)
plt.legend()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot([x[0] for x in pred_states])
plt.plot([x[0] for x in real_states])
plt.title('First state component')
plt.subplot(222)
plt.plot([x[1] for x in pred_states])
plt.plot([x[1] for x in real_states])
plt.title('Second state component')
plt.subplot(223)
plt.plot([x[2] for x in pred_states])
plt.plot([x[2] for x in real_states])
plt.title('Third state component')
plt.subplot(224)
plt.plot([x[3] for x in pred_states])
plt.plot([x[3] for x in real_states])
plt.title('Forth state component')
plt.show()

