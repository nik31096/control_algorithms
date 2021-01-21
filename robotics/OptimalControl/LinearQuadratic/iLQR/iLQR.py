import numpy as np
import autograd.numpy as autonp
from autograd import grad, jacobian
import matplotlib.pyplot as plt

from robotics.OptimalControl.LinearQuadratic.iLQR.LQR import LQR_TV
from robotics.OptimalControl.LinearQuadratic.neural_iLQR.NNDynamics import NonLinearDynamics

from tqdm import trange
import pickle as pkl
import os


class iLQR:
    def __init__(self,
                 dynamics,
                 initial_state=None,
                 cost=None,
                 state_dim=None,
                 action_dim=None,
                 final_state=None,
                 horizon=100,
                 **kwargs):

        if isinstance(dynamics, NonLinearDynamics):
            self.f = dynamics.dyn
            self.f_x = dynamics.derivative(arg=0)
            self.f_u = dynamics.derivative(arg=1)
            self.state_dim = dynamics.state_dim
            self.action_dim = dynamics.action_dim
        else:
            self.f = dynamics
            self.f_x = jacobian(self.f, 0)
            self.f_u = jacobian(self.f, 1)
            self.state_dim = state_dim
            self.action_dim = action_dim

        if cost is None:
            self.g = self.cost
        else:
            self.g = cost
        self.g_x = grad(self.g, 0)
        self.g_u = grad(self.g, 1)
        self.g_xx = jacobian(self.g_x, 0)
        self.g_uu = jacobian(self.g_u, 1)

        self.lqr = LQR_TV(T=horizon)
        self.T = horizon
        if final_state is None:
            self.final_state = autonp.zeros((self.state_dim,))
        else:
            self.final_state = autonp.array(final_state)
        if initial_state is None:
            self.initial_state = autonp.zeros((self.state_dim,))
        else:
            self.initial_state = initial_state

        self.Q_f = np.diag(np.array(np.block([1e5*np.ones((self.state_dim,)), 1])))
        self.k_gains = kwargs.get('k_gains', [])
        self.controller_k_gains = self.k_gains

    def set_final_state(self, final_state, inv_kin):
        final_state = inv_kin(final_state[0], final_state[1])
        self.final_state = np.array([final_state[0], 0, final_state[1], 0])

    def _backward_pass(self, states, controls):
        A_seq, B_seq, Q_seq, R_seq = [], [], [], []
        for t, (x, u) in enumerate(zip(states, controls)):
            f_x = self.f_x(x, u)
            f_u = self.f_u(x, u)
            # print(f"f_x.shape = {f_x.shape}, f_u.shape = {f_u.shape}", end=', ')
            g_x = self.g_x(x, u, x, u, self.final_state)
            # print(f"g_x.shape = {g_x.shape}", end=', ')
            g_xx = self.g_xx(x, u, x, u, self.final_state)
            # print(f"g_xx.shape = {g_xx.shape}", end=', ')
            g_u = self.g_u(x, u, x, u, self.final_state)
            # print(f"g_u.shape = {g_u.shape}", end=', ')
            g_uu = self.g_uu(x, u, x, u, self.final_state)
            # print(f"g_uu.shape = {g_uu.shape}")
            g = 0.5*self.g(x, u, x, u, self.final_state) * np.ones((1, 1))

            A = np.block([
                [f_x, np.zeros((f_x.shape[0], 1))],
                [np.zeros((1, f_x.shape[1])), np.ones((1, 1))]
            ])

            B = np.block([
                [f_u, np.zeros((f_u.shape[0], 1))],
                [np.zeros((1, f_u.shape[1])), np.zeros((1, 1))]
            ])

            Q = np.block([
                [g_xx, np.zeros((g_x.shape[0], 1))],
                [g_x, g]
            ])

            R = np.block([
                [g_uu, np.zeros((self.action_dim, 1))],
                [g_u[np.newaxis, :], g]
            ])

            A_seq.append(A)
            B_seq.append(B)
            Q_seq.append(Q)
            R_seq.append(R)

        self.lqr.set_parameters(A_seq, B_seq, Q_seq, R_seq, x_T=np.block([self.final_state, 0]), Q_f=self.Q_f)
        k_gains = self.lqr.get_k_gains()

        return k_gains

    def _forward_pass(self, k_gains):
        new_states = [self.initial_state]
        new_controls = []
        x = self.initial_state
        for t, k in enumerate(k_gains):
            u = np.dot(-k, np.block([x - self.final_state, 1]))[:-1]
            x = self.f(x - self.final_state, u) + self.final_state
            new_states.append(x)
            new_controls.append(u)

        return new_states, new_controls

    def _get_states(self, controls):
        x = self.initial_state
        states = [x]
        for t in range(self.T):
            x = self.f(x - self.final_state, controls[t]) + self.final_state
            states.append(x)

        return states

    def fit_controller(self, controls=None, epochs=20, initial_state=None, verbose=0):
        assert epochs > 0, "Epochs should be positive integers."
        if controls is None:
            controls = [0.001 * autonp.ones((self.action_dim,)) for _ in range(self.T)]

        if initial_state is not None:
            self.initial_state = autonp.array(initial_state)

        states = self._get_states(controls)
        epoch_gen = range(epochs) if verbose == 0 else trange(epochs)
        for epoch in epoch_gen:
            k_gains = self._backward_pass(states, controls)
            states, controls = self._forward_pass(k_gains)

        self.k_gains = k_gains

        return controls

    def get_control(self, x, t, evaluate=False):
        if evaluate:
            control = np.dot(-self.controller_k_gains[t], np.block([x - self.final_state, 1]))[:-1]
        else:
            k = self.k_gains[t]
            control = np.dot(-k, np.block([x - self.final_state, 1]))[:-1]
            self.controller_k_gains.append(k)

        return control

    def save_controller(self, filename):
        controller_params = {
            'k_gains': self.controller_k_gains,
            'final_state': self.final_state,
            'dyn': self.f,
            'T': self.T,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

        if not os.path.exists('./saved_controllers'):
            os.mkdir('./saved_controllers')

        with open(f'./saved_controllers/{filename}.ilqr', 'wb') as f:
            pkl.dump(controller_params, f)

    @staticmethod
    def load_controller(filename):
        with open(filename, 'rb') as f:
            controller_params = pkl.load(f)

        controller = iLQR(dynamics=controller_params['dyn'],
                          initial_state=controller_params['initial_state'],
                          final_state=controller_params['final_state'],
                          state_dim=controller_params['state_dim'],
                          action_dim=controller_params['action_dim'],
                          horizon=controller_params['T'],
                          k_gains=controller_params['k_gains'],
                          controls=controller_params['controls'])

        return controller

    @staticmethod
    def cost(x, u, x_i, u_i, x_target):
        alpha = 0.97
        pure_cost = 0.5 * autonp.sum(u ** 2)
        cost_correction = autonp.sum((x - x_i) ** 2) + autonp.sum((u - u_i) ** 2)
        c = alpha * cost_correction + (1 - alpha) * pure_cost

        return c


class iLQR_v1(iLQR):
    def __init__(self,
                 dynamics,
                 initial_state=None,
                 cost=None,
                 state_dim=None,
                 action_dim=None,
                 final_state=None,
                 horizon=100,
                 **kwargs):
        super(iLQR_v1, self).__init__(dynamics, initial_state, cost, state_dim,
                                      action_dim, final_state, horizon, **kwargs)


if __name__ == '__main__':
    def dynamics(x, u):
        """
        Nonlinear dynamics example: pendulum with gravity
        :param x: vector of [rotation angle, rotation speed]
        :param u: scalar control that is just torque for pendulum control
        :return: next state based on the equation of motion solution
        """
        dt = 0.01
        m = 1
        l = 0.5
        g = 10
        f = autonp.array([x[1], u[0] / (m * l ** 2) - g * autonp.sin(x[0])])
        next_state = x + dt * f
        return next_state


    def cost(x, u, x_i, u_i, x_target):
        alpha = 0.97
        pure_cost = autonp.sum(5 * u ** 2)
        cost_correction = autonp.sum((x - x_i) ** 2) + autonp.sum((u - u_i) ** 2)
        c = alpha * cost_correction + (1 - alpha) * pure_cost

        return c


    horizon = 400
    epochs = 20
    initial_state = autonp.array([3, 1], dtype=autonp.float32)

    ilqr = iLQR(dynamics=dynamics, cost=cost, state_dim=2, action_dim=1,
                initial_state=initial_state, horizon=horizon)
    ilqr.fit_controller(epochs=20, initial_state=initial_state, verbose=1)

    state = initial_state
    states, controls = [state], []
    for t in range(horizon):
        control = ilqr.get_control(state, t)
        next_state = dynamics(state, control)
        state = next_state

        controls.append(control)
        states.append(state)

    print("Final state is:", states[-1], ", final action is:", controls[-1])
    plt.figure(figsize=(20, 20))
    plt.subplot(311)
    plt.plot([x[0] for x in states])
    plt.title("First state component")
    plt.subplot(312)
    plt.plot([x[1] for x in states])
    plt.title("Second state component")
    plt.subplot(313)
    plt.plot([u[0] for u in controls])
    plt.title("Action")
    plt.show()
