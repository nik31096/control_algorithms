import gym

from robotics.SAC.sac_agent import SACAgent

env = gym.make('FetchReach-v1')
state = env.reset()

agent = SACAgent(env.observation_space['observation'].shape[0],
                 env.observation_space['achieved_goal'].shape[0],
                 env.action_space.shape[0],
                 [env.action_space.low[0], env.action_space.high[0]],
                 gamma=0.99,
                 tau=0.001,
                 q_lr=1e-4,
                 value_lr=1e-4,
                 policy_lr=1e-4,
                 )
agent.load_pretrained_models('sac_2')

while True:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

    state = next_state


