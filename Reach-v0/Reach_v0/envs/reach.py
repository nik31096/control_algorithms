import os
import numpy as np
import autograd.numpy as autonp
from copy import deepcopy

import matplotlib.pyplot as plt
import time

import gym
from gym.utils import seeding, EzPickle
from gym.envs.robotics import utils
import mujoco_py
import glfw

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


DEFAULT_SIZE = 84


class Reach2DEnv(gym.GoalEnv):
    def __init__(self, 
                 model_path,
                 n_actions, 
                 n_substeps,
                 distance_threshold,
                 initial_qpos,
                 state_form='image',
                 width=84,
                 height=84,
                 dynamical_goal=True,
                 limited_goal_area=True,
                 has_obstacle=False):
        self.n_substeps = n_substeps

        self.state_form = state_form
        self.width = width
        self.height = height
        self.dynamical_goal = dynamical_goal
        self.limited_goal_area = limited_goal_area
        self.has_obstacle = has_obstacle

        self.action_space = gym.spaces.Box(-1., 1., shape=(n_actions, ), dtype='float32')

        if state_form == 'image':
            self.observation_space = gym.spaces.Dict(dict(
                observation=gym.spaces.Box(0., 1., shape=(self.width, self.height, 3), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')))
        elif state_form == 'angles':
            self.observation_space = gym.spaces.Dict(dict(
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')))

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

        # ReachEnv part
        self.distance_threshold = distance_threshold

        self.max_u = 5

        # TODO: add automatic link length determination from xml file
        d = 0.03  # radius of the link capsule
        self.l1 = 1.2
        self.l2 = 1
        r1 = self.l1 / 2
        r2 = self.l2 / 2
        m1 = 3
        m2 = 3
        I1 = m1 * (self.l1 ** 2 + 3 * d ** 2) / 12
        I2 = m2 * (self.l2 ** 2 + 3 * d ** 2) / 12

        self.alpha = float(I1 + I2 + m1 * r1 ** 2 + m2 * (self.l1 ** 2 + r2 ** 2))
        self.beta = float(m2 * self.l1 * r2)
        self.delta = float(I2 + m2 * r2 ** 2)

        self.seed()
        self._env_setup()
        self.initial_state = deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    @staticmethod
    def goal_distance(achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape, "Goal shapes are different!"
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        return d

    def set_goal(self, goal_x, goal_y):
        self.goal = np.array([goal_x, goal_y, self.goal[2]])
        self._render_callback()

    def set_pose(self, theta1, theta2):
        self.sim.data.set_joint_qpos('joint_1', theta1)
        self.sim.data.set_joint_qpos('joint_2', theta2)

        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

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
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.cam.distance = 5
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
            self._viewers[mode] = self.viewer

        if mode == 'human':
            self.viewer.render()
        elif mode == 'rgb_array':
            img = self.viewer._read_pixels_as_in_window(resolution=(width, height))

            return img
        else:
            raise NameError("No such mode is available")

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
        # TODO: get rid of achieved_goal in the state formulation.
        end_pos = self.sim.data.get_site_xpos('end_site')
        if self.state_form == 'image':
            # camera_image = self.render(mode='rgb_array')
            camera_image = self.sim.render(width=self.width, height=self.height, mode='offscreen',
                                           camera_name='upper_camera', depth=False)
            camera_image = np.transpose(camera_image[::-1, :, :], [2, 0, 1]) / 255
            # camera_image = np.transpose(camera_image, [2, 0, 1]) / 255

            return camera_image, end_pos

        elif self.state_form == 'angles':
            theta1 = self.sim.data.get_joint_qpos('joint_1')
            theta1_dot = self.sim.data.get_joint_qvel('joint_1')
            theta2 = self.sim.data.get_joint_qpos('joint_2')
            theta2_dot = self.sim.data.get_joint_qvel('joint_2')

            state = np.array([theta1, theta1_dot, theta2, theta2_dot])

            return state, end_pos

    def get_next_state(self, state, action):
        '''
        Forward kinamatics method written with autograd package to support derivatives for optimal control.
        '''
        # state = state['observation']
        #  dims      0          1         2         3
        # state = [theta1, theta1_dot, theta2, theta2_dot]

        theta1 = state[:, 0] if len(state.shape) >= 2 else state[0]
        theta1_dot = state[:, 1] if len(state.shape) >= 2 else state[1]
        theta2 = state[:, 2] if len(state.shape) >= 2 else state[2]
        theta2_dot = state[:, 3] if len(state.shape) >= 2 else state[3]

        tau1 = self.max_u*action[:, 0] if len(action.shape) >= 2 else self.max_u*action[0]
        tau2 = self.max_u*action[:, 1] if len(action.shape) >= 2 else self.max_u*action[1]

        a1 = self.alpha + 2*self.beta*autonp.cos(theta2)
        a2 = self.delta + self.beta*autonp.cos(theta2)
        a3 = self.delta
        a4 = self.beta*autonp.sin(theta2)

        k1 = 3
        k2 = 3

        f1 = theta1_dot
        f2 = (-k1*theta1_dot + tau1 + a4*theta2_dot*(2*theta1_dot + theta2_dot) - a2 / a3 * (tau2 - k2*theta2_dot - a4*theta1_dot**2)) / (a1 + a2**2 / a3)
        f3 = theta2_dot
        f4 = (-k2*theta2_dot + tau2 - a2 * (tau1 - k1*theta1_dot + a4*theta2_dot*(2*theta1_dot + theta2_dot) - a2 / a3 * (tau2 - k2*theta2_dot - a4*theta1_dot**2)) /
              (a1 + a2**2 / a3) - a4*theta1_dot**2) / a3
        if len(state.shape) >= 2:
            f = autonp.transpose(autonp.array([f1, f2, f3, f4], dtype='float'), [1, 0])
        else:
            f = autonp.array([f1, f2, f3, f4], dtype='float')
        next_state = state + self.dt * f

        return next_state

    def forward_kinematics(self, theta1, theta2):
        x = self.l1 * autonp.cos(theta1) + self.l2 * autonp.cos(theta1 + theta2)
        y = self.l1 * autonp.sin(theta1) + self.l2 * autonp.sin(theta1 + theta2)
        return np.array([x, y])

    def inverse_kinematics(self, x, y):
        x, y = y, -x
        r = np.sqrt(x**2 + y**2)
        if x > 0 and y > 0:
            phi = np.arctan(y / x)
        elif x > 0 and y < 0:
            phi = np.arctan(y / x) + 2*np.pi
        elif x < 0:
            phi = np.arctan(y / x) + np.pi
        else:
            raise NotImplementedError("This case of coordinates is not considered")

        theta1 = phi - np.arccos(np.clip((r**2 + self.l1**2 - self.l2**2) / (2*r*self.l1), a_min=-1, a_max=1))
        theta2 = np.arccos(np.clip((r**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2), a_min=-1, a_max=1))

        return theta1, theta2

    def _env_setup(self):
        for part, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(part, value)

        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

    def _sample_goal(self):
        # TODO: rewrite goal generation considering constrained env
        end_pos = self.sim.data.get_site_xpos('end_site')
        assert end_pos.shape == (3, )
        # Sample goal in the circle around origin with radius = self.l1 + self.l2
        r_end = self.l1 + self.l2
        theta_end = np.arccos(end_pos[0] / r_end)
        goal_z = end_pos[2]
        # if the constrained env then limit the angle
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
        # but for now obstacle pos is static, see manipulator_with_obstacle.xml file, line 27-29
        # obstacle_pos = np.array([1.2, 1.2, 0])


class ReachEnv_v0(Reach2DEnv, EzPickle):
    """
    Environment contains 2 link robot arm without joint limits and observation is 256x256x3 image
    """
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
                            state_form='image',
                            dynamical_goal=True,
                            limited_goal_area=True)
        EzPickle.__init__(self)


class ReachEnv_v1(Reach2DEnv, EzPickle):
    """
    Environment contains 2 link robot arm with joint limits and observation is
    4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components
    """
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
                            limited_goal_area=True
                            )
        EzPickle.__init__(self)


class ReachEnv_v2(Reach2DEnv, EzPickle):
    """
    Environment contains 2 link robot arm without joint limits and observation is
    4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components
    """
    def __init__(self):
        model_path = 'manipulator_without_joint_limits.xml'
        n_actions = 2
        n_substeps = 20
        distance_threshold = 0.05

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='angles',
                            limited_goal_area=False,
                            dynamical_goal=True)
        EzPickle.__init__(self)


class ReachEnv_v3(Reach2DEnv, EzPickle):
    """
    Environment contains 2 link robot arm without joint limits and observation is
    4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components
    Number of substeps is 1 and the length of the episode is 2000 timesteps,
    which is equal to 1*2000 = 2000 - length of the rest of environments
    """
    def __init__(self):
        model_path = 'manipulator_without_joint_limits.xml'
        n_actions = 2
        n_substeps = 1
        distance_threshold = 0.08

        Reach2DEnv.__init__(self,
                            model_path=model_path,
                            n_actions=n_actions,
                            n_substeps=n_substeps,
                            distance_threshold=distance_threshold,
                            initial_qpos={'joint_1': 0, 'joint_2': 0},
                            state_form='angles',
                            limited_goal_area=False,
                            dynamical_goal=False
                            )
        EzPickle.__init__(self)


class ReachEnv_v4(Reach2DEnv, EzPickle):
    """
    Environment contains 2 link robot arm without joint limits and observation is
    4d vector which consists of (theta1, theta1_dot, theta2, theta2_dot) components.
    Also there is an obstacle which is blue heavy cylinder standing vertically.
    """
    def __init__(self):
        # TODO: Add the cylider mentioned in the docs!!!
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
