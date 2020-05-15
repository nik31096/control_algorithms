import gym
import Reach_v0

from robotics.SAC.sac_agent import SACAgent_v1

env = gym.make('Reach-v1')
state = env.reset()

agent = SACAgent_v1(env.observation_space['observation'].shape[0],
                    env.observation_space['achieved_goal'].shape[0],
                    env.action_space.shape[0],
                    [env.action_space.low[0], env.action_space.high[0]],
                    gamma=0.99,
                    tau=0.001,
                    q_lr=1e-4,
                    alpha=1,
                    alpha_lr=1e-4,
                    policy_lr=1e-4,
                    image_as_state=False
)
agent.load_pretrained_models('sac_4')

while True:
    #action = agent.select_action(state, evaluate=True)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

    state = next_state


