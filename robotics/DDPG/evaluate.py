import gym

from robotics.DDPG.ddpg_agent import DDPG

env = gym.make('FetchReach-v1')
state = env.reset()

agent = DDPG(env.observation_space['observation'].shape[0],
             env.observation_space['achieved_goal'].shape[0],
             env.action_space.shape[0],
             [env.action_space.low[0], env.action_space.high[0]],
             gamma=0.99,
             tau=0.001,
             actor_lr=1e-4,
             critic_lr=1e-3
             )
agent.load_pretrained("./weights", 'ddpg_1')

while True:
    action = agent.select_action(state, mode='eval')
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

    state = next_state


