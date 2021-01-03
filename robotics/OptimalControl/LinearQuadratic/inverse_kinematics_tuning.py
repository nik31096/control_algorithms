import Reach_v0
import gym

from matplotlib import pyplot as plt

env = gym.make('Reach-v3')
state = env.reset()

goal = state['desired_goal'][:-1]
print(goal)

img1 = env.render(mode='rgb_array')

final_state = env.inverse_kinematics(goal[0], goal[1])
print(final_state)
env.set_pose(final_state[0], final_state[1])

img2 = env.render(mode='rgb_array')

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(img2)
plt.show()

