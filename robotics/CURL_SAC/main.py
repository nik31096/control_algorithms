import gym
import Reach_v0

import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.CURL_SAC.curl_sac_agent import CURL_SACAgent
from robotics.ExperienceReplays import HindsightExperienceReplay
from robotics.CURL_SAC.utils import ImageBuffer
from multiprocessing_environment.subproc_env import SubprocVecEnv


def main(device):
    # hyperparameters and same code snippets for both modes
    n_epochs = 5000
    n_substeps = 10
    gamma = 0.999
    tau = 5e-3
    batch_size = 128
    hidden_dim = 10
    model_name = "reach_image_2"
    writer_name = f"./runs/{model_name}"

    writer = SummaryWriter(writer_name)

    def make_env(env_id):

        def _f():
            env = gym.make(env_id)
            return env

        return _f

    env_id = "Reach-v0"
    n_envs = 32

    envs = [make_env(env_id) for _ in range(n_envs)]
    envs = SubprocVecEnv(envs, context='fork', in_series=4)
    states = envs.reset()

    test_env = gym.make(env_id)
    n_steps = test_env._max_episode_steps
    env_params = {'obs': hidden_dim,
                  'actions': test_env.action_space.shape[0],
                  'goals': test_env.observation_space['achieved_goal'].shape[0],
                  'reward_function': test_env.compute_reward,
                  'max_episode_timesteps': n_steps}

    img_buf = ImageBuffer(size=10000, device=device)
    img_buf.put(states)

    agent = CURL_SACAgent(hidden_dim=hidden_dim,
                          goal_dim=envs.observation_space["achieved_goal"].shape[0],
                          action_dim=envs.action_space.shape[0],
                          action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                          gamma=gamma,
                          tau=tau,
                          alpha=1,
                          q_lr=3e-4,
                          alpha_lr=3e-4,
                          policy_lr=3e-4,
                          device=device)

    replay_buffer = HindsightExperienceReplay(env_params=env_params,
                                              size=1000000,
                                              n_envs=n_envs,
                                              use_achieved_goal=True,
                                              k=8)

    pretrained = False
    if pretrained:
        agent.load_pretrained_models('reach_1')

    epoch_delay = 20

    for epoch in trange(n_epochs):
        for step in range(n_steps):
            encoded_states = agent.encode_obs(states, to_numpy=True)
            actions = agent.select_action(encoded_states)
            next_states, rewards, dones, info = envs.step(actions)
            encoded_next_states = agent.encode_obs(next_states, to_numpy=True)
            img_buf.put(next_states['observation'])
            replay_buffer.collect_episodes(encoded_states, actions, rewards, encoded_next_states, dones)
            # Training
            if epoch > epoch_delay:
                # CURL training
                for inner_step in range(n_substeps):
                    obs_batch = img_buf.sample(batch_size=256)
                    contrastive_loss = agent.train_encoder(obs_batch)
                    writer.add_scalar("Contrastive_loss", contrastive_loss,
                                      n_envs*(epoch * n_steps * n_substeps + step*n_substeps + inner_step))
                # RL training
                batch = replay_buffer.sample(batch_size)
                q_1_loss, q_2_loss, policy_loss, mean_q, entropy_loss, alpha = agent.train(batch, update_alpha=True)
                # logging
                writer.add_scalar("Q1_loss", q_1_loss, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Q2_loss", q_2_loss, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Policy_loss", policy_loss, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Mean_Q", mean_q, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Entropy loss", entropy_loss, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Alpha", alpha, n_envs * (epoch * n_steps + step))
                writer.add_scalar("Success_rate", round(sum([_info['is_success'] for _info in info]) / n_envs, 2),
                                  n_envs * (epoch * n_steps + step))

            states = next_states
            if np.all(dones):
                states = envs.reset()
                replay_buffer.store_episodes()
                break

        ep2log = 20
        if (epoch + 1) % ep2log == 0 and epoch > epoch_delay:
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


if __name__ == '__main__':
    main(device='cuda')
