import gym
import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange

from .ddpg_multi_env import DDPG
from .utils import ExperienceReplay
from .multiprocessing_environment.subproc_env import SubprocVecEnv


# hyperparameters and same code snippets for both modes
gamma = 0.95
EPS = 0.01
batch_size = 512
writer_name = "./runs/run_3"

replay_buffer = ExperienceReplay()
writer = SummaryWriter(writer_name)

mode = 'multi_env'

if mode == 'multi_env':

    def make_env(env_id):
        def _f():
            env = gym.make(env_id)
            return env
        return _f


    env_id = "FetchReach-v1"
    n_envs = 8

    distance_writers = [SummaryWriter(f'{writer_name}/distance_writer.env_{i}') for i in range(n_envs)]

    envs = [make_env(env_id) for _ in range(n_envs)]
    envs = SubprocVecEnv(envs, context='fork', in_series=1)
    states = envs.reset()

    agent = DDPG(observation_space_shape=envs.observation_space["observation"].shape[0],
                 goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                 action_space_shape=envs.action_space.shape[0],
                 action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                 gamma=gamma,
                 writer=writer
                 )
    pretrained = True
    if pretrained:
        agent.load_pretrained('./weights', 'ddpg_1')

    for epoch in trange(1000):
        for step in range(1000):
            actions = agent.select_action(states['observation'], states['desired_goal'], states['achieved_goal'])
            next_states, rewards, dones, info = envs.step(actions.data.numpy())
            replay_buffer.put(states, actions, rewards, next_states, dones)
            for i in range(n_envs):
                if dones[i]:
                    distance = np.linalg.norm(states['desired_goal'][i] - states['achieved_goal'][i])
                    envs.reset_env(env_index=i)
                    distance_writers[i].add_scalar("Desired-achieved distance", distance, epoch)
            states = next_states

            if len(replay_buffer) > batch_size:
                # Training
                batch = replay_buffer.sample(batch_size)
                agent.train(batch, epoch, iterations=10 if len(replay_buffer) < 100000 else 100)
        if (epoch + 1) % 50 == 0:
            agent.save_models('./weights', 'ddpg_1')

    replay_buffer.save('./buffer')

else:
    env = gym.make('FetchReach-v1')
    state = env.reset()

    agent = DDPG(observation_space_shape=env.observation_space["observation"].shape[0],
                 goal_space_shape=env.observation_space["achieved_goal"].shape[0],
                 action_space_shape=env.action_space.shape[0],
                 action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                 gamma=gamma,
                 writer=writer,
                 mode='single_env'
                 )
    state = env.reset()
    for epoch in trange(100000):
        for step in range(1000):
            action = agent.select_action(state['observation'], state['desired_goal'], state['achieved_goal'])
            next_state, reward, done, _ = env.step(action.data.numpy())
            replay_buffer.put(state, action, reward, next_state, done)
            # TODO: add success rate instead of episode reward
            if done:
                writer.add_scalar("Success rate", 1 if reward == -0.0 else 0, epoch)
                state = env.reset()
                break

            state = next_state

            # Training
            batch = replay_buffer.sample(3000)
            agent.train(batch, epoch=epoch, iterations=10)

        if (epoch + 1) % 50 == 0:
            agent.save_models('./weights', 'single_env_ddpg_1')
