import autograd.numpy as np
from autograd import jacobian, grad

from tqdm import trange


class DDP:
    """
    Heavily based on this implementation: https://github.com/neka-nat/ilqr-gym
    """
    def __init__(self, a1, a2, a3, dynamics_model, forward_dynamics, umax, state_dim, pred_time=50):
        self.alpha = 1
        self.mu = 1e-6
        self.mu_min = 1e-6
        self.sigma_0 = 2
        self.sigma = 2
        self.c1 = 0.01

        self.pred_time = pred_time
        self.state_dim = state_dim
        self.umax = umax
        self.goal = np.zeros(2)
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.lf = lambda x: 0.5*np.sum(a1*np.square(self.goal - forward_dynamics(x[0], x[2])) +
                                       a2*np.square(x[1]) + a3*np.square(x[3]))
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)

        self.l = lambda x, u: 0.5*np.sum(np.square(u))
        self.l_x = grad(self.l, 0)
        self.l_u = grad(self.l, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)

        self.f = dynamics_model
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def set_goal(self, goal):
        self.goal = goal

    @staticmethod
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def increase_mu(self):
        print("Increase mu", end=": ")
        print("old mu = ", self.mu, end='; ')
        self.sigma = max(self.sigma_0, self.sigma*self.sigma_0)
        self.mu = max(self.mu_min, self.mu*self.sigma)
        print("new mu = ", self.mu)

    def decrease_mu(self):
        print("Decrease mu", end=": ")
        print("old mu = ", self.mu, end='; ')
        self.sigma = min(1 / self.sigma_0, self.sigma / self.sigma_0)
        self.mu = self.mu*self.sigma if self.mu*self.sigma >= self.mu_min else 0
        print("new mu = ", self.mu)

    def backward(self, x_seq, u_seq):
        print("[INFO] Start backward pass")
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        K_seq = []

        delta_J = 0

        for t in trange(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.dot(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.dot(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.dot(np.dot(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              np.einsum("i,ijk->jk", self.v_x[t + 1], self.f_xx(x_seq[t], u_seq[t]))
            tmp = np.dot(f_u_t.T, self.v_xx[t + 1])
            tmp_tilda = np.dot(f_u_t.T, self.v_xx[t + 1] + self.mu*np.eye(self.state_dim))
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.dot(tmp, f_u_t) + \
                   np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_uu(x_seq[t], u_seq[t]))
            q_uu_tilda = self.l_uu(x_seq[t], u_seq[t]) + np.dot(tmp_tilda, f_u_t) + \
                         np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_uu(x_seq[t], u_seq[t]))
            pos_def = self.is_pos_def(q_uu_tilda)
            if not pos_def:
                self.increase_mu()
                restart = True
                return None, None, restart, None

            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.dot(tmp, f_x_t) + \
              np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))
            q_ux_tilda = self.l_ux(x_seq[t], u_seq[t]) + np.dot(tmp, f_x_t) + \
                         np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))
            inv_q_uu = np.linalg.inv(q_uu)
            inv_q_uu_tilda = np.linalg.inv(q_uu_tilda)
            k = -np.dot(inv_q_uu_tilda, q_u)
            K = -np.dot(inv_q_uu, q_ux_tilda)
            dv = 0.5*np.dot(k.T, np.dot(q_uu, k)) + np.dot(k.T, q_u)
            self.v[t] += dv
            # - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_x[t] = q_x + np.dot(K.T, np.dot(q_uu, k)) + np.dot(K.T, q_u) + np.dot(q_ux.T, k)
            # q_xx + np.matmul(q_ux.T, K)
            self.v_xx[t] = q_xx + np.dot(K.T, np.dot(q_uu, K)) + np.dot(K.T, q_ux) + np.dot(q_ux.T, K)
            k_seq.append(k)
            K_seq.append(K)

            delta_J += self.alpha*np.dot(k.T, q_u) + self.alpha**2*0.5*np.dot(k.T, np.dot(q_uu, k))
        k_seq.reverse()
        K_seq.reverse()

        self.decrease_mu()
        restart = False

        return k_seq, K_seq, restart, delta_J

    def get_cost(self, initial_state, controls):
        J = self.l(initial_state, controls[0])
        state = initial_state
        for i in trange(1, len(controls) - 1):
            state = self.f(state, controls[i - 1])
            J += self.l(state, controls[i])

        J += self.lf(state)

        return J

    def forward(self, x_seq, u_seq, k_seq, kk_seq, delta_J):
        print("[INFO] Start forward pass")
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = self.alpha*k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            new_control = u_seq[t] + control
            u_seq_hat[t] = np.clip(new_control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])

        old_cost = self.get_cost(x_seq[0], u_seq)
        new_cost = self.get_cost(x_seq[0], u_seq_hat)
        print("old_cost", old_cost)
        print("new cost", new_cost)
        print("delta J", delta_J)
        z = (old_cost - new_cost) / abs(delta_J)

        if z < self.c1:
            restart = True
            self.alpha = self.alpha*0.999 if self.alpha > 1e-3 else 1e-3

            print("z = ", z, ", alpha = ", self.alpha)
            return None, None, restart

        return x_seq_hat, u_seq_hat, False

    def plan(self, states, actions, n_epochs):
        for _ in trange(n_epochs):
            k, K = self.backward(x_seq=states, u_seq=actions)
            states, actions = self.forward(x_seq=states, u_seq=actions, k_seq=k, kk_seq=K)

        return actions
