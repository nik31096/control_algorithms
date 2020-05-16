import gym
import Reach_v0

import numpy as np
from tensorboardX import SummaryWriter

from tqdm import trange
import os

from robotics.TD3.td3_agent import TD3Agent
from robotics.TD3.utils import ExperienceReplay, HindsightExperienceReplay
from multiprocessing_environment.subproc_env import SubprocVecEnv

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)


def main(mode, device):
    # hyperparameters and same code snippets for both modes
    n_epochs = 500000
    gamma = 0.999
    tau = 5e-3
    batch_size = 64
    model_name = "pick_td3_2"
    writer_name = f"./runs/{model_name}"

    writer = SummaryWriter(writer_name)

    if mode == 'multi_env':

        def make_env(env_id):
            def _f():
                env = gym.make(env_id)
                return env
            return _f

        env_id = "FetchPickAndPlace-v1"
        n_envs = 16

        envs = [make_env(env_id) for _ in range(n_envs)]
        envs = SubprocVecEnv(envs, context='fork', in_series=2)
        states = envs.reset()

        test_env = gym.make(env_id)
        env_params = {'obs': test_env.observation_space['observation'].shape[0],
                      'actions': test_env.action_space.shape[0],
                      'goals': test_env.observation_space['achieved_goal'].shape[0],
                      'reward_function': test_env.compute_reward,
                      'max_episode_timesteps': test_env._max_episode_steps}

        replay_buffer = HindsightExperienceReplay(env_params=env_params, size=20000, n_envs=n_envs, k=8)

        agent = TD3Agent(observation_space_shape=envs.observation_space["observation"].shape[0],
                         goal_space_shape=envs.observation_space["achieved_goal"].shape[0],
                         action_space_shape=envs.action_space.shape[0],
                         action_ranges=(envs.action_space.low[0], envs.action_space.high[0]),
                         gamma=gamma,
                         tau=tau,
                         q_lr=1e-3,
                         policy_lr=1e-3,
                         device=device,
                         image_as_state=False
                         )

        pretrained = True
        if pretrained:
            agent.load_pretrained_models('pick_td3_1')

        for epoch in trange(n_epochs):
            for step in range(1000):
                iteration = epoch * env_params['max_episode_timesteps'] + step
                if epoch < 500:
                    actions = np.array([test_env.action_space.sample() for _ in range(n_envs)])
                else:
                    actions = agent.select_action(states)
                next_states, rewards, dones, info = envs.step(actions)
                # replay_buffer.put(states, actions, rewards, next_states, dones)
                replay_buffer.collect_episodes(states, actions, rewards, next_states, dones)
                # Training
                if epoch > 10:
                    # Training
                    batch = replay_buffer.sample(batch_size)
                    agent.train(batch, iteration, writer)

                states = next_states
                if np.all(dones):
                    states = envs.reset()
                    replay_buffer.store_episodes()
                    writer.add_scalar("Success_rate", round(sum([_info['is_success'] for _info in info]) / n_envs, 3),
                                      epoch)
                    break

            ep2log = 100
            if (epoch + 1) % ep2log == 0:
                agent.save_models(model_name)
                if not os.path.exists('./figures'):
                    os.mkdir('./figures')
                # testing
                success = 0
                for _ in range(10):
                    state = test_env.reset()
                    rewards_sum = 0
                    distance = 1000
                    for _ in range(1000):
                        action = agent.select_action(state, evaluate=True)
                        next_state, reward, done, info = test_env.step(action)
                        rewards_sum += reward
                        distance = min(distance, np.linalg.norm(state['desired_goal'] - state['achieved_goal'], axis=-1))
                        if done:
                            if info['is_success']:
                                success += 1
                            writer.add_scalar("Evaluation distance", distance, global_step=(epoch + 1) // ep2log)
                            writer.add_scalar("Episode reward sum", rewards_sum, global_step=(epoch + 1) // ep2log)
                            # final_state_from_upper_camera = np.transpose(test_env.render(mode='rgb_array'), [2, 0, 1])
                            # writer.add_image("Final_state_from_upper_camera", final_state_from_upper_camera,
                            #                 global_step=(epoch + 1) // ep2log)
                            break
                writer.add_scalar("Test success rate", round(success / 10, 2), global_step=(epoch + 1) // ep2log)

    else:
        replay_buffer = ExperienceReplay(size=1000000, mode=mode, device=device)

        env = gym.make('Reach-v1')
        state = env.reset()

        agent = TD3Agent(observation_space_shape=env.observation_space["observation"].shape[0],
                         goal_space_shape=env.observation_space["achieved_goal"].shape[0],
                         action_space_shape=env.action_space.shape[0],
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
                # replay_buffer.collect_episodes(state, actions, rewards, next_states, dones)
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
