import gym

env = gym.make('FetchReach-v1')

print(env.observation_space)
print(env.action_space)

state = env.reset()

print(state)


# done = False
# while not done:
#     action = env.action_space.sample()
#     print(action)
#     next_s, r, done, _ = env.step(action)
#     env.render()
#     if done:
#         break
#     state = next_s

