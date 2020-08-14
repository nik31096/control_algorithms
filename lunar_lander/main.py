import gym

import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from lunar_lander.agents import DDPGAgent
from lunar_lander.utils import ExperienceReplay
from multiprocessing_environment.subproc_env import SubprocVecEnv


def main(mode, device):
    # hyperparameters and same code snippets for both modes
    n_epochs = 50000
    gamma = 0.999
    tau = 5e-3
    batch_size = 256
    model_name = "lander_1"
    writer_name = f"./runs/{model_name}"

    writer = SummaryWriter(writer_name)

    if mode == 'multi_env':

        def make_env(env_id):
            def _f():
                env = gym.make(env_id)
                return env
            return _f

        env_id = "LunarLanderContinuous-v2"
        n_envs = 48

        envs = [make_env(env_id) for _ in range(n_envs)]
        envs = SubprocVecEnv(envs, context='fork', in_series=6)
        states = envs.reset()

        test_env = gym.make(env_id)

        replay_buffer = ExperienceReplay(size=int(1e7 / n_envs))

        agent = DDPGAgent(observation_space_shape=envs.observation_space.shape[0],
                          action_space_shape=envs.action_space.shape[0],
                          action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                          gamma=gamma,
                          tau=tau,
                          q_lr=3e-4,
                          policy_lr=3e-4,
                          device=device,
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
                replay_buffer.put(states, actions, rewards, next_states, dones)
                # Training
                if epoch > epoch_delay:
                    # Training
                    batch = replay_buffer.sample(batch_size)
                    # entropy_loss, alpha
                    q_1_loss, policy_loss, mean_q = agent.train(batch)

                    writer.add_scalar("Q1_loss", q_1_loss, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Policy_loss", policy_loss, epoch*test_env._max_episode_steps + step)
                    writer.add_scalar("Mean_Q", mean_q, epoch*test_env._max_episode_steps + step)

                states = next_states
                if np.all(dones):
                    states = envs.reset()
                    break

            ep2log = 50
            if (epoch + 1) % ep2log == 0 and epoch > epoch_delay:
                agent.save_models(model_name)
                # testing
                state = test_env.reset()
                rewards_sum = 0
                for _ in range(1000):
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = test_env.step(action)
                    rewards_sum += reward
                    if done:
                        writer.add_scalar("Episode reward sum", rewards_sum, global_step=(epoch + 1) // ep2log)
                        break

    else:
        replay_buffer = ExperienceReplay(size=1000000, mode=mode, device=device)

        env = gym.make('Reach-v1')
        state = env.reset()

        agent = DDPGAgent(observation_space_shape=env.observation_space["observation"].shape[0],
                            action_space_shape=env.action_space.shape[0],
                            action_ranges=(env.action_space.low[0], env.action_space.high[0]),
                            gamma=gamma,
                            tau=tau,
                            q_lr=1e-4,
                            policy_lr=1e-4,
                            device=device,
                            mode='single_env',
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
                            q_1_loss, q_2_loss, policy_loss, entropy_loss, alpha = agent.train(batch)
                        else:
                            q_1_loss, q_2_loss, policy_loss = agent.train(batch)
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
