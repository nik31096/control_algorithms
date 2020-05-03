import gym
import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.SAC.sac_agent import SACAgent_v1, SACAgent_v0
from robotics.SAC.utils import ExperienceReplay, DistanceLogging
from multiprocessing_environment.subproc_env import SubprocVecEnv


# hyperparameters and same code snippets for both modes
n_epochs = 500000
gamma = 0.99
tau = 5e-3
batch_size = 512
writer_name = "./runs/run_5"
distance_writer_name = "run_5"

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

    test_env = gym.make(env_id)

    agent = SACAgent_v1(observation_space_shape=envs.observation_space["observation"].shape[0],
                        goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                        action_space_shape=envs.action_space.shape[0],
                        action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                        gamma=gamma,
                        tau=tau,
                        q_lr=3e-4,
                        alpha_lr=1e-4,
                        policy_lr=3e-4
                        )

    pretrained = False
    if pretrained:
        agent.load_pretrained_models('sac_1')

    for epoch in trange(n_epochs):
        for step in range(1000):
            actions = agent.select_action(states)
            next_states, rewards, dones, info = envs.step(actions)
            replay_buffer.put(states, actions, rewards, next_states, dones)
            states = next_states
            if np.all(dones):
                states = envs.reset()
                if len(replay_buffer) > batch_size:
                    # Training
                    batch = replay_buffer.sample(batch_size)
                    q_1_loss, q_2_loss, policy_loss, entropy_loss, alpha = agent.train(batch)

                    writer.add_scalar("Q1_loss", q_1_loss, epoch)
                    writer.add_scalar("Q2_loss", q_2_loss, epoch)
                    writer.add_scalar("Policy_loss", policy_loss, epoch)
                    writer.add_scalar("Entropy loss", entropy_loss, epoch)
                    writer.add_scalar("Alpha", alpha, epoch)

                    if (epoch + 1) % 10000 == 0:
                        distance_logger.calculate_distances(next_states)
                break

        '''
        for i in range(n_envs):
            if dones[i]:
                distance = np.linalg.norm(states['desired_goal'][i] - states['achieved_goal'][i])
                state = envs.reset_env(env_index=i)
                next_states['observation'][i] = state['observation']
                next_states['desired_goal'][i] = state['desired_goal']
                next_states['achieved_goal'][i] = state['achieved_goal']
                distance_logger.put(index=i, value=distance)
        
        
        if len(replay_buffer) > 10*batch_size:
            # Training
            n_iters = 10 if len(replay_buffer) < 100000 else 100
            for iter_ in range(n_iters):
                batch = replay_buffer.sample(batch_size)
                value_loss, policy_loss = agent.train(batch)

            writer.add_scalar("Value_loss", value_loss, epoch)
            writer.add_scalar("Policy_loss", policy_loss, epoch)
        '''

        if (epoch + 1) % 10000 == 0:
            distance_logger.get_plot(distance_writer_name)
            agent.save_models('sac_4')

        if (epoch + 1) % 500 == 0:
            if not os.path.exists('./figures'):
                os.mkdir('./figures')
            # testing
            state = test_env.reset()
            while True:
                action = agent.select_action(state, mode='eval')
                next_state, reward, done, _ = test_env.step(action)
                if done:
                    distance = np.linalg.norm(state['desired_goal'] - state['achieved_goal'])
                    writer.add_scalar("Evaluation distance", distance, global_step=(epoch + 1) // 500)
                    break

    replay_buffer.save('./buffer')

else:
    env = gym.make('FetchReach-v1')
    state = env.reset()

    agent = SACAgent_v0(observation_space_shape=env.observation_space["observation"].shape[0],
                        goal_space_shape=env.observation_space["achieved_goal"].shape[0],
                        action_space_shape=env.action_space.shape[0],
                        action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                        gamma=gamma,
                        tau=tau,
                        q_lr=1e-4,
                        value_lr=1e-4,
                        policy_lr=1e-4,
                        train_device='cuda',
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
            agent.train(batch)

        if (epoch + 1) % 50 == 0:
            agent.save_models('./weights', 'single_env_ddpg_1')
