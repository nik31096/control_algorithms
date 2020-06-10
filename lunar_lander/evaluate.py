import gym

from lunar_lander.agents import DDPGAgent

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='ddpg_1')
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--render', type=bool, default=False)

args = parser.parse_args()

env = gym.make('LunarLanderContinuous-v2')

agent = DDPGAgent(env.observation_space.shape[0],
                  env.action_space.shape[0],
                  [env.action_space.low[0], env.action_space.high[0]],
                  gamma=0.99,
                  tau=0.001,
                  q_lr=1e-4,
                  policy_lr=1e-4,
                  device='cpu'
                  )
agent.load_pretrained_models(args.model_name, evaluate=True)

n_runs = args.n_runs
reward_sum = 0
for _ in range(n_runs):
    done = False
    state = env.reset()
    i = 1

    while not done:
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, info = env.step(action)
        reward_sum += reward
        if args.render:
            env.render()
        if done:
            break

        state = next_state
        i += 1

print(f"Number of test runs: {n_runs}, average reward: {round(reward_sum / n_runs, 3)}")


