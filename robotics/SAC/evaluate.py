import gym
import Reach_v0

import matplotlib.pyplot as plt

from robotics.SAC.sac_agent import SACAgent_v1

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchReach-v1')
parser.add_argument('--model_name', type=str, default='sac_1')
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--render', type=bool, default=False)

args = parser.parse_args()

env = gym.make(args.env_name)

agent = SACAgent_v1(env.observation_space['observation'].shape[0],
                    env.observation_space['achieved_goal'].shape[0],
                    env.action_space.shape[0],
                    [env.action_space.low[0], env.action_space.high[0]],
                    gamma=0.99,
                    tau=0.001,
                    q_lr=1e-4,
                    policy_lr=1e-4,
                    device='cpu',
                    alpha=1.0,
                    alpha_lr=1e-4
                    )
agent.load_pretrained_models(args.model_name, evaluate=True)

states, actions = [], []
n_runs = args.n_runs
success = 0
for _ in range(n_runs):
    done = False
    state = env.reset()
    states.append(state['observation'])
    i = 1
    while not done:
        action = agent.select_action(state, evaluate=True)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        if args.render:
            env.render()
        if done:
            if info['is_success']:
                success += 1

        state = next_state
        i += 1

    plt.figure(figsize=(30, 30))
    plt.subplot(321)
    plt.plot([x[0] for x in states], label='First joint angle')
    plt.subplot(322)
    plt.plot([x[1] for x in states], label='Second joint angle')
    plt.subplot(323)
    plt.plot([x[2] for x in states], label='First joint angle velocity')
    plt.subplot(324)
    plt.plot([x[3] for x in states], label='Second joint angle velocity')
    plt.subplot(325)
    plt.plot([a[0] for a in actions], label='First joint torque')
    plt.subplot(326)
    plt.plot([a[1] for a in actions], label='Second joint torque')


print(f"Number of test runs: {n_runs}, success rate: {round(success / n_runs, 3)}")


