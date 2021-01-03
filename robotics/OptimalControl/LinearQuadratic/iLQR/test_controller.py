import gym
from matplotlib import pyplot as plt

from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR


ilqr = iLQR.load_controller('../saved_controllers/3_epochs.lqr')

env = gym.make('Reach-v3')
state = env.reset()
env.set_goal(goal_x=ilqr.final_state[0], goal_y=ilqr.final_state[2])

t1, t2 = [state['observation'][0]], [state['observation'][2]]
controls = []

for t in range(2000):
    control = ilqr.get_control(state['observation'], t)
    next_state, reward, done, _ = env.step(control)
    controls.append(control)
    t1.append(next_state['observation'][0])
    t2.append(next_state['observation'][2])
    state = next_state
    env.render()

print("Real final state:", state["observation"])

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.plot(t1)
plt.title("Theta 1")
plt.subplot(222)
plt.plot(t2)
plt.title("Theta 2")
plt.subplot(223)
plt.plot([a[0] for a in controls])
plt.title("First joint torque")
plt.subplot(224)
plt.plot([a[1] for a in controls])
plt.title("Second joint torque")
plt.show()
