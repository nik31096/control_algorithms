import Reach_v0
import gym

from robotics.OptimalControl.LinearQuadratic.iLQR.iLQR import iLQR
from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import Dynamics


env = gym.make('Reach-v2')
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]
dynamics = Dynamics(state_dim=state_dim, action_dim=action_dim)
ilqr = iLQR(dynamics=dynamics)
ilqr.load_controller('../saved_controllers/')