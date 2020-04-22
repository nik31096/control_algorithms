import gym
import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.SAC.sac_agent import SACAgent
from robotics.SAC.utils import ExperienceReplay, DistanceLogging
from multiprocessing_environment.subproc_env import SubprocVecEnv


# hyperparameters and same code snippets for both modes
n_epochs = 1000000
gamma = 0.99
tau = 1e-3
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

    distance_logger = DistanceLogging(n_envs)

    envs = [make_env(env_id) for _ in range(n_envs)]
    envs = SubprocVecEnv(envs, context='fork', in_series=1)
    states = envs.reset()

    agent = SACAgent(observation_space_shape=envs.observation_space["observation"].shape[0],
                     goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                     action_space_shape=envs.action_space.shape[0],
                     action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                     gamma=gamma,
                     tau=tau,

                     )
    '''
    agent = DDPG(observation_space_shape=envs.observation_space["observation"].shape[0],
                 goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                 action_space_shape=envs.action_space.shape[0],
                 action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                 gamma=gamma,
                 tau=tau,
                 actor_lr=1e-4,
                 critic_lr=1e-3
                 )
    '''
    pretrained = True
    if pretrained:
        agent.load_pretrained('./weights', 'ddpg_1')

    for epoch in trange(n_epochs):
        actions = agent.select_action(states)
        next_states, rewards, dones, info = envs.step(actions)
        replay_buffer.put(states, actions, rewards, next_states, dones)
        for i in range(n_envs):
            if dones[i]:
                distance = np.linalg.norm(states['desired_goal'][i] - states['achieved_goal'][i])
                state = envs.reset_env(env_index=i)
                next_states['observation'][i] = state['observation']
                next_states['desired_goal'][i] = state['desired_goal']
                next_states['achieved_goal'][i] = state['achieved_goal']
                distance_logger.put(index=i, value=distance)

        states = next_states

        if len(replay_buffer) > batch_size:
            # Training
            n_iters = 10 if len(replay_buffer) < 100000 else 100
            for iter_ in range(n_iters):
                batch = replay_buffer.sample(batch_size)
                value_loss, policy_loss = agent.train(batch)

            writer.add_scalar("Value_loss", value_loss, epoch)
            writer.add_scalar("Policy_loss", policy_loss, epoch)

            agent.syncronize_online_networks()

        if (epoch + 1) % 100000 == 0:
            distance_logger.get_plot(writer_name)

        if (epoch + 1) % 500 == 0:
            if not os.path.exists('./figures'):
                os.mkdir('./figures')
            agent.save_models('./weights', 'ddpg_1')
            # testing
            env = gym.make(env_id)
            state = env.reset()
            while True:
                action = agent.select_action(state, mode='eval')
                next_state, reward, done, _ = env.step(action.data.numpy())
                if done:
                    distance = np.linalg.norm(state['desired_goal'] - state['achieved_goal'])
                    writer.add_scalar("Evaluation distance", distance, global_step=(epoch + 1) // 500)
                    break
            del env

    replay_buffer.save('./buffer')

else:
    env = gym.make('FetchReach-v1')
    state = env.reset()

    agent = DDPG(observation_space_shape=env.observation_space["observation"].shape[0],
                 goal_space_shape=env.observation_space["achieved_goal"].shape[0],
                 action_space_shape=env.action_space.shape[0],
                 action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                 gamma=gamma,
                 tau=tau,
                 mode='single_env'
                 )

    for epoch in trange(n_epochs):
        for step in range(1000):
            action = agent.select_action(state)
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
