import os
import numpy as np
import autograd.numpy as autonp
from copy import deepcopy

import gym
from gym.utils import seeding, EzPickle
from gym.envs.robotics import utils
import mujoco_py

from pprint import pprint


DEFAULT_SIZE = 500


class Reach2DEnv(gym.GoalEnv):
    def __init__(self, 
                 model_path,
                 n_actions, 
                 n_substeps,
                 distance_threshold,
                 initial_qpos,
                 mode='model-free',
                 state_form='image',
                 width=256,
                 height=256,
                 dynamical_goal=True,
                 limited_goal_area=True,
                 has_obstacle=False
                 ):
        '''
        :param mode: two options: 'model-free' for model-free RL and 'model-based' for optimal control (OptimalControl and DDP)
        '''
        self.n_substeps = n_substeps

        self.mode = mode
        self.state_form = state_form
        self.width = width
        self.height = height
        self.dynamical_goal = dynamical_goal
        self.limited_goal_area = limited_goal_area
        self.has_obstacle = has_obstacle

        self.action_space = gym.spaces.Box(-1., 1., shape=(n_actions, ), dtype='float32')

        if mode == 'model-free' and state_form == 'image':
            self.observation_space = gym.spaces.Dict(dict(
                observation=gym.spaces.Box(0., 1., shape=(self.width, self.height, 3), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
            ))
        elif mode == 'model-free' and state_form == 'angles':
            self.observation_space = gym.spaces.Dict(dict(
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
            ))

        # MuJoCo part
        if model_path.startswith('/'):
            full_path = model_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), 'assets', model_path)

        if not os.path.exists(full_path):
            raise IOError(f"File {full_path} does not exist")

        model = mujoco_py.load_model_from_path(full_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)

        self.initial_qpos = initial_qpos
        self.viewer = None
        self._viewers = {}

        self.seed()
        self._env_setup()
        self.initial_state = deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        # ReachEnv part
        self.distance_threshold = distance_threshold

        self.max_u = 3

        d = 0.03  # radius of the capsule
        l1 = 1
        l2 = 1
        r1 = l1 / 2
        r2 = l2 / 2
        m1 = 3
        m2 = 3
        I1 = m1 * (l1 ** 2 + 3 * d ** 2) / 12
        I2 = m2 * (l2 ** 2 + 3 * d ** 2) / 12

        self.alpha = float(I1 + I2 + m1 * r1 ** 2 + m2 * (l1 ** 2 + r2 ** 2))
        self.beta = float(m2 * l1 * r2)
        self.delta = float(I2 + m2 * r2 ** 2)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    @staticmethod
    def goal_distance(achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape, "Goal shapes are different!"
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        return d

    def set_goal(self, goal):
        self.goal = goal

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = self.goal_distance(achieved_goal, desired_goal)
        return -(d > self.distance_threshold).astype(np.float32)

    # Env methods

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # to do render, save images in folder and execute
        # ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
        self._render_callback()
        self.viewer = self._viewers.get(mode)
        if mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer.cam.distance = 5
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -45
                self._viewers[mode] = self.viewer

            self.viewer.render()
        elif mode == 'rgb_array':
            image = self.sim.render(width=512, height=512, mode='offscreen',
                                    camera_name='upper_camera', depth=False)
            return image[::-1, :, :]
        else:
            print("No such mode is available")
            raise NameError

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('goal')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    def _set_action(self, action):
        assert len(action.shape) == 1
        action = action.copy()  # ensure that we don't change the action outside of this scope
        for i in range(action.shape[0]):
            self.sim.data.ctrl[i] = self.max_u*action[i]

    def step(self, action):
        self._set_action(action)
        self.sim.step()
        obs, achieved_goal = self._get_obs()
        state = {'observation': obs, 'achieved_goal': achieved_goal, "desired_goal": self.goal}

        done = False
        info = {'is_success': self._is_success(achieved_goal, self.goal)}
        reward = self.compute_reward(achieved_goal, self.goal, info)

        return state, reward, done, info

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        return True

    def reset(self):
        super(Reach2DEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        if self.dynamical_goal:
            self.goal = self._sample_goal().copy()
        self._render_callback()
        obs, end_pos = self._get_obs()
        state = {'observation': obs, 'achieved_goal': end_pos, "desired_goal": self.goal}

        return state
    
    def _get_obs(self):
        end_pos = self.sim.data.get_site_xpos('end_site')
        if self.state_form == 'image':
            camera_image = self.sim.render(width=self.width, height=self.height, mode='offscreen',
                                           camera_name='upper_camera', depth=False)
            camera_image = np.transpose(camera_image[::-1, :, :], [2, 0, 1]) / 255

            return camera_image, end_pos

        elif self.state_form == 'angles':
            theta1 = self.sim.data.get_joint_qpos('joint_1')
            theta1_dot = self.sim.data.get_joint_qvel('joint_1')
            theta2 = self.sim.data.get_joint_qpos('joint_2')
            theta2_dot = self.sim.data.get_joint_qvel('joint_2')

            state = np.array([theta1, theta1_dot, theta2, theta2_dot])

            return state, end_pos

    def _get_next_state(self, state, action):
        # state = state['observation']
        action = self.max_u*action
        #  dims      0          1         2         3
        # state = [theta1, theta1_dot, theta2, theta2_dot]
        a1 = self.alpha + 2*self.beta*autonp.cos(state[2])
        a2 = self.delta + self.beta*autonp.cos(state[2])
        a3 = self.beta*autonp.sin(state[2])
        a4 = self.delta
        denom = a1*a4 - a2*a2

        f1 = state[1]
        f2 = a4 / denom * (action[0] - a2 * action[1] / a4 + a2*a3*state[1]*state[1] / a4 +
                           a3*state[3]*(2*state[1] + state[2]))
        f3 = state[3]
        f4 = action[1] / a4 - a2 / denom * (action[0] - a2 * action[1] / a4 + a2*a3*state[1]*state[1] / a4 +
                                            a3*state[3]*(2*state[1] + state[2]) - a3 * state[1] * state[1] / a4)
        f = autonp.array([f1, f2, f3, f4], dtype='float')

        next_state = state + self.dt * f

        # next_state = np.clip(next_state, [-3, next_state[1], -3, next_state[3]], [3, next_state[1], 3, next_state[3]])
        return next_state

    def get_next_state(self, state, action):
        for i in range(self.n_substeps):
            next_state = self._get_next_state(state, action)
            state = next_state

        return next_state

    def _env_setup(self):
        for part, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(part, value)

        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

    def _sample_goal(self):
        end_pos = self.sim.data.get_site_xpos('end_site')
        assert end_pos.shape == (3, )
        # Sample goal in the area around arm initial point with radius = 2*each_arm_len
        r_end = np.sqrt(end_pos[0]**2 + end_pos[1]**2)
        theta_end = np.arccos(end_pos[0] / r_end)
        goal_z = end_pos[2]
        theta_limit = 0.25*np.pi if self.limited_goal_area else 0.95*np.pi
        r_limit_left = 1 if self.limited_goal_area else 0.1
        r_limit_right = 1.8

        theta = self.np_random.uniform(theta_end - theta_limit, theta_end + theta_limit)
        r = self.np_random.uniform(r_limit_left, r_limit_right)
        goal_x = r*np.cos(theta)
        goal_y = r*np.sin(theta)
        goal = np.array([goal_x, goal_y, goal_z])

        return goal.copy()

    def _sample_obstacle(self):
        end_pos = self.sim.data.get_site_xpos('end_site')
        goal = self.goal.copy()
        # TODO: figure out how to place obstacle to ensure the reachability by manipulator,
        #  but for now obstacle pos is static, see manipulator_wuth_obstacle.xml file, line 27-29
        # obstacle_pos = np.array([1.2, 1.2, 0])


class ReachEnv_v0(Reach2DEnv, EzPickle):
    '''
        Environment contains 2 link robot arm with joint limits and observation is 256x256x3 image
    '''
    def __init__(self):
        model_path = 'manipulator.xml'
        n_actions = 2
        n_substeps = 30
        distance_threshold = 0.03

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='image'
                            )
        EzPickle.__init__(self)


class ReachEnv_v1(Reach2DEnv, EzPickle):
    '''
        Environment contains 2 link robot arm with joint limits and observation is
        4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components
    '''
    def __init__(self):
        model_path = 'manipulator.xml'
        n_actions = 2
        self.n_substeps = 20
        distance_threshold = 0.1

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=self.n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='angles',
                            limited_goal_area=False
                            )
        EzPickle.__init__(self)


class ReachEnv_v2(Reach2DEnv, EzPickle):
    '''
        Environment contains 2 link robot arm without joint limits and observation is
        4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components
    '''
    def __init__(self):
        model_path = 'manipulator_without_joint_limits.xml'
        n_actions = 2
        n_substeps = 20
        distance_threshold = 0.08

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='angles',
                            limited_goal_area=False,
                            dynamical_goal=True
                            )
        EzPickle.__init__(self)


class ReachEnv_v3(Reach2DEnv, EzPickle):
    '''
        Environment contains 2 link robot arm without joint limits and observation is
        4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components.
        Also there is an obstacle which is blue heavy cylinder standing vertically.
    '''
    def __init__(self):
        model_path = 'manipulator_with_obstacle.xml'
        n_actions = 2
        n_substeps = 20
        distance_threshold = 0.1

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='angles',
                            limited_goal_area=False,
                            has_obstacle=True
                            )
        EzPickle.__init__(self)
