import Reach_v0
import gym

from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR
from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import NonLinearDynamics

from matplotlib import pyplot as plt


env = gym.make('Reach-v3')
max_steps = env._max_episode_steps
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]
ilqr = iLQR.load_controller('./saved_controllers/one_step_mpc.ilqr')
final_state = (ilqr.final_state[0], ilqr.final_state[2])
cartesian_final_state = env.forward_kinematics(theta1=final_state[0], theta2=final_state[1])
env.set_goal(goal_x=cartesian_final_state[0], goal_y=cartesian_final_state[1])

state = env.reset()
states = [state['observation']]
actions = []
for t in range(max_steps):
    u = ilqr.get_control(state['observation'], t, evaluate=True)
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
