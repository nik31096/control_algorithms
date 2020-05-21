from gym.envs.registration import register

register(
    id='Reach-v0',
    entry_point='Reach_v0.envs:ReachEnv_v0',
    max_episode_steps=100,
)

register(
    id='Reach-v1',
    entry_point='Reach_v0.envs:ReachEnv_v1',
    max_episode_steps=100,
)

register(
    id='Reach-v2',
    entry_point='Reach_v0.envs:ReachEnv_v2',
    max_episode_steps=100,
)

register(
    id='Reach-v3',
    entry_point='Reach_v0.envs:ReachEnv_v3',
    max_episode_steps=100,
)
