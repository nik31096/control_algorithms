import gym
import Reach_v0

import numpy as np
import torch
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.SAC.sac_agent import SACAgent_v1
from robotics.SAC.utils import ExperienceReplay, HindsightExperienceReplay
from multiprocessing_environment.subproc_env import SubprocVecEnv

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


def main(mode, device):
    # hyperparameters and same code snippets for both modes
    n_epochs = 50000
    gamma = 0.999
    tau = 5e-3
    batch_size = 64
    model_name = "reach_obstacle_1"
    writer_name = f"./runs/{model_name}"

    writer = SummaryWriter(writer_name)

    if mode == 'multi_env':

        def make_env(env_id):
            def _f():
                env = gym.make(env_id)
                return env
            return _f

        env_id = "Reach-v4"
        n_envs = 48

        envs = [make_env(env_id) for _ in range(n_envs)]
        envs = SubprocVecEnv(envs, context='fork', in_series=6)
        states = envs.reset()

        test_env = gym.make(env_id)
        env_params = {'obs': test_env.observation_space['observation'].shape[0],
                      'weights': test_env.action_space.shape[0],
                      'goals': test_env.observation_space['achieved_goal'].shape[0],
                      'reward_function': test_env.compute_reward,
                      'max_episode_timesteps': test_env._max_episode_steps}

        replay_buffer = HindsightExperienceReplay(env_params=env_params, size=int(1e7 / n_envs), n_envs=n_envs,
                                                  use_achieved_goal=True, k=8)

        agent = SACAgent_v1(observation_space_shape=envs.observation_space["observation"].shape[0],
                            goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                            action_space_shape=envs.action_space.shape[0],
                            action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                            gamma=gamma,
                            tau=tau,
                            alpha=1,
                            q_lr=3e-4,
                            alpha_lr=3e-4,
                            policy_lr=3e-4,
                            device=device
                            )

        pretrained = False
        if pretrained:
            agent.load_pretrained_models('reach_1')

        epoch_delay = 50

        for epoch in trange(n_epochs):
            for step in range(1000):
                if epoch < epoch_delay:
                    actions = np.array([envs.action_space.sample() for _ in range(n_envs)])
                else:
                    actions = agent.select_action(states)
                next_states, rewards, dones, info = envs.step(actions)
                # replay_buffer.put(states, weights, rewards, next_states, dones)
                replay_buffer.collect_episodes(states, actions, rewards, next_states, dones)
                # Training
                if epoch > epoch_delay:
                    # Training
                    batch = replay_buffer.sample(batch_size)
                    # entropy_loss, alpha
                    q_1_loss, q_2_loss, policy_loss, mean_q, entropy_loss, alpha = agent.train(batch, update_alpha=True)

                    writer.add_scalar("Q1_loss", q_1_loss, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Q2_loss", q_2_loss, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Policy_loss", policy_loss, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Mean_Q", mean_q, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Entropy loss", entropy_loss, epoch)
                    writer.add_scalar("Alpha", alpha, epoch)
                    writer.add_scalar("Success_rate", round(sum([_info['is_success'] for _info in info]) / n_envs, 2),
                                      epoch*test_env._max_episode_steps + step)

                states = next_states
                if np.all(dones):
                    states = envs.reset()
                    replay_buffer.store_episodes()
                    break

            ep2log = 50
            if (epoch + 1) % ep2log == 0 and epoch > epoch_delay:
                agent.save_models(model_name)
                if not os.path.exists('./figures'):
                    os.mkdir('./figures')
                # testing
                state = test_env.reset()
                rewards_sum = 0
                distance = 1000
                for _ in range(1000):
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = test_env.step(action)
                    rewards_sum += reward
                    distance = min(np.linalg.norm(state['desired_goal'] - state['achieved_goal']), distance)
                    if done:
                        writer.add_scalar("Evaluation distance", distance, global_step=(epoch + 1) // ep2log)
                        writer.add_scalar("Episode reward sum", rewards_sum, global_step=(epoch + 1) // ep2log)
                        final_state_from_upper_camera = np.transpose(test_env.render(mode='rgb_array'), [2, 0, 1])
                        writer.add_image("Final_state_from_upper_camera", final_state_from_upper_camera,
                                         global_step=(epoch + 1) // ep2log)
                        break

    else:
        replay_buffer = ExperienceReplay(size=1000000, mode=mode, device=device)

        env = gym.make('Reach-v1')
        state = env.reset()

        agent = SACAgent_v1(observation_space_shape=env.observation_space["observation"].shape[0],
                            goal_space_shape=env.observation_space["achieved_goal"].shape[0],
                            action_space_shape=env.action_space.shape[0],
                            action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                            gamma=gamma,
                            tau=tau,
                            alpha=1,
                            q_lr=1e-4,
                            alpha_lr=1e-4,
                            policy_lr=1e-4,
                            mode='single_env'
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
                        update_alpha = False
                        if update_alpha:
                            q_1_loss, q_2_loss, policy_loss, entropy_loss, alpha = agent.train(batch, update_alpha=True)
                        else:
                            q_1_loss, q_2_loss, policy_loss = agent.train(batch, update_alpha=False)
                        writer.add_scalar("Q1_loss", q_1_loss, epoch)
                        writer.add_scalar("Q2_loss", q_2_loss, epoch)
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
