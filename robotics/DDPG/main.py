import gym
import Reach_v0

import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.DDPG.ddpg_agent import DDPGAgent
from robotics.ExperienceReplays import HindsightExperienceReplay, ExperienceReplay
from multiprocessing_environment.subproc_env import SubprocVecEnv


def main(mode, device):
    # hyperparameters and same code snippets for both modes
    n_epochs = 5000
    gamma = 0.99
    tau = 5e-2
    batch_size = 64
    model_name = "reachV2run_1"
    writer_name = f"./runs/{model_name}"

    writer = SummaryWriter(writer_name)

    if mode == 'multi_env':

        def make_env(env_id):
            def _f():
                env = gym.make(env_id)
                return env
            return _f

        env_id = "Reach-v2"
        n_envs = 32

        envs = [make_env(env_id) for _ in range(n_envs)]
        envs = SubprocVecEnv(envs, context='fork', in_series=4)
        states = envs.reset()

        test_env = gym.make(env_id)
        n_steps = test_env._max_episode_steps
        env_params = {'obs': test_env.observation_space['observation'].shape[0],
                      'weights': test_env.action_space.shape[0],
                      'goals': test_env.observation_space['achieved_goal'].shape[0],
                      'reward_function': test_env.compute_reward,
                      'max_episode_timesteps': n_steps}

        replay_buffer = HindsightExperienceReplay(env_params=env_params, size=1000000, n_envs=n_envs, k=16,
                                                  use_achieved_goal=False)

        agent = DDPGAgent(observation_dim=envs.observation_space["observation"].shape[0],
                          goal_dim=envs.observation_space["achieved_goal"].shape[0],
                          action_dim=envs.action_space.shape[0],
                          action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                          gamma=gamma,
                          tau=tau,
                          q_lr=3e-4,
                          policy_lr=3e-4,
                          device=device,
                          image_as_state=False)

        pretrained = False
        if pretrained:
            agent.load_pretrained_models('ddpg_1')

        for epoch in trange(n_epochs):
            for step in range(n_steps):
                iteration = n_envs*(epoch*n_steps + step)
                actions = agent.select_action(states, noise=True, evaluate=False)
                next_states, rewards, dones, info = envs.step(actions)
                replay_buffer.collect_episodes(states, actions, rewards, next_states, dones)
                states = next_states

                if epoch > 200:
                    # Training
                    batch = replay_buffer.sample(batch_size)
                    q_loss, mean_q, policy_loss = agent.train(batch)

                    writer.add_scalar("Q1_loss", q_loss, iteration)
                    writer.add_scalar("Mean_Q", mean_q, iteration)
                    writer.add_scalar("Policy_loss", policy_loss, iteration)

                if np.all(dones):
                    states = envs.reset()
                    replay_buffer.store_episodes()
                    writer.add_scalar("Success_rate", sum([_info['is_success'] for _info in info]) / n_envs,
                                      n_envs * epoch * n_steps)
                    break

            ep2log = 100
            if (epoch + 1) % ep2log == 0:
                agent.save_models(model_name)
                if not os.path.exists('./figures'):
                    os.mkdir('./figures')
                # testing
                success = 0
                rewards_sum = 0
                for _ in range(10):
                    state = test_env.reset()
                    for _ in range(n_steps):
                        action = agent.select_action(state, evaluate=True)
                        next_state, reward, done, info = test_env.step(action)
                        rewards_sum += reward
                        if done:
                            if info['is_success']:
                                success += 1
                            break

                writer.add_scalar("Test_average_rewards", rewards_sum / 10, n_envs * epoch * n_steps)
                writer.add_scalar("Test_success_rate", round(success / 10, 5), n_envs * epoch * n_steps)

    else:
        replay_buffer = ExperienceReplay(size=1000000, mode=mode, device=device)

        env = gym.make('Reach-v1')
        state = env.reset()

        agent = DDPGAgent(observation_dim=env.observation_space["observation"].shape[0],
                          goal_dim=env.observation_space["achieved_goal"].shape[0],
                          action_dim=env.action_space.shape[0],
                          action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                          gamma=gamma,
                          tau=tau,
                          q_lr=1e-4,
                          policy_lr=1e-4,
                          device=device,
                          mode='single_env',
                          image_as_state=False
                          )

        for epoch in trange(n_epochs):
            for step in range(1000):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                replay_buffer.put(state, action, reward, next_state, done)
                # replay_buffer.collect_episodes(state, weights, rewards, next_states, dones)
                state = next_state
                if done:
                    # replay_buffer.store_episodes()
                    state = env.reset()
                    if len(replay_buffer) > batch_size:
                        # Training
                        batch = replay_buffer.sample(batch_size)
                        q_1_loss, policy_loss = agent.train(batch)
                        writer.add_scalar("Q1_loss", q_1_loss, epoch)
                        writer.add_scalar("Policy_loss", policy_loss, epoch)
                        if (epoch + 1) % 500 == 0:
                            distance = np.linalg.norm(state['desired_goal'] - state['achieved_goal'])
                            writer.add_scalar("Evaluation distance", distance, global_step=(epoch + 1) // 500)
                            writer.add_scalar("Success", info['is_success'], global_step=(epoch + 1) // 500)

                    break

            if (epoch + 1) % 10000 == 0:
                agent.save_models('sac_8')


if __name__ == '__main__':
    main(mode='multi_env', device='cuda')
