import gym
import Reach_v0

from robotics.DDPG.ddpg_agent import DDPGAgent

from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchReach-v1')
parser.add_argument('--model_name', type=str, default='ddpg_1')
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--render', type=bool, default=False)

args = parser.parse_args()

env = gym.make(args.env_name)

agent = DDPGAgent(env.observation_space['observation'].shape[0],
                  env.observation_space['achieved_goal'].shape[0],
                  env.action_space.shape[0],
                  [env.action_space.low[0], env.action_space.high[0]],
                  gamma=0.99,
                  tau=0.001,
                  q_lr=1e-4,
                  policy_lr=1e-4,
                  image_as_state=False,
                  device='cpu'
                  )
agent.load_pretrained_models(args.model_name, evaluate=True)
state = env.reset()

n_runs = args.n_runs
success = 0
for _ in range(n_runs):
    done = False
    state = env.reset()
    i = 1
    while not done:
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        if args.render:
            env.render()
        if done:
            if info['is_success']:
                success += 1

        state = next_state
        i += 1

print(f"Number of test runs: {n_runs}, success rate: {round(success / n_runs, 3)}")


