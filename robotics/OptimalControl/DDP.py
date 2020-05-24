import autograd.numpy as np
from autograd import jacobian, grad
from tqdm import trange


class DDP_v1:
    """
    Heavily based on this implementation: https://github.com/neka-nat/ilqr-gym
    This variant of DifferentialDynamicProgramming is from paper: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    """
    def __init__(self, a1, a2, a3, dynamics_model, forward_dynamics, umax, state_dim, pred_time=50):
        self.alpha = 1
        self.mu = 1e-6
        self.mu_min = 1e-6
        self.sigma_0 = 2
        self.sigma = 2
        self.c1 = 0.5
        self.deltaJ = []

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

        beta = 0.9
        # 0.5*np.sum(np.square(u))
        # self.l = lambda x, u, x_nom=0, u_nom=0: (1 - beta)*np.sum(1e-12*np.cosh(30*u)) + \
        #                                         beta*(np.linalg.norm(x_nom - x)**2 + np.linalg.norm(u_nom - x)**2)
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

    def backward(self, x_seq, u_seq, iter_number):
        print("[INFO] Start backward pass, iteration:", iter_number)
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        K_seq = []
        self.deltaJ = []

        for t in trange(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t], ) + np.dot(f_x_t.T, self.v_x[t + 1])
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
            eig_values, eig_vectors = np.linalg.eig(q_uu_tilda)
            if not np.all(eig_values > 0):
                self.increase_mu()
                restart = True
                return None, None, restart

            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.dot(tmp, f_x_t) + \
              np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))
            q_ux_tilda = self.l_ux(x_seq[t], u_seq[t]) + np.dot(tmp_tilda, f_x_t) + \
                         np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))

            inv_q_uu_tilda = np.dot(eig_vectors, np.dot(np.diag(1.0 / eig_values), eig_vectors.T))
            k = -np.dot(inv_q_uu_tilda, q_u)
            K = -np.dot(inv_q_uu_tilda, q_ux_tilda)
            dv = 0.5*np.dot(k.T, np.dot(q_uu, k)) + np.dot(k.T, q_u)
            self.v[t] += dv
            # - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_x[t] = q_x + np.dot(K.T, np.dot(q_uu, k)) + np.dot(K.T, q_u) + np.dot(q_ux.T, k)
            # q_xx + np.matmul(q_ux.T, K)
            self.v_xx[t] = q_xx + np.dot(K.T, np.dot(q_uu, K)) + np.dot(K.T, q_ux) + np.dot(q_ux.T, K)
            k_seq.append(k)
            K_seq.append(K)

            self.deltaJ.append((np.dot(k.T, q_u), 0.5*np.dot(k.T, np.dot(q_uu, k))))

        k_seq.reverse()
        K_seq.reverse()

        self.decrease_mu()
        restart = False

        return k_seq, K_seq, restart

    def get_cost(self, initial_state, controls):
        J = self.l(initial_state, controls[0])
        state = initial_state
        for i in range(1, len(controls) - 1):
            next_state = self.f(state, controls[i - 1])
            J += self.l(next_state, controls[i])
            state = next_state

        J += self.lf(state)

        return J

    def compute_deltaJ(self, alpha):
        ans = sum([alpha*x[0] + alpha**2*x[1] for x in self.deltaJ])

        return ans

    def forward(self, x_seq, u_seq, k, K, iter_number):
        print("[INFO] Start forward pass, iteration number:", iter_number)
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = self.alpha*k[t] + np.matmul(K[t], x_seq_hat[t] - x_seq[t])
            new_control = u_seq[t] + control
            u_seq_hat[t] = np.clip(new_control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])

        # print(u_seq_hat)
        old_cost = self.get_cost(x_seq[0], u_seq)
        new_cost = self.get_cost(x_seq[0], u_seq_hat)
        delta_J = self.compute_deltaJ(self.alpha)
        print("old_cost", old_cost)
        print("new cost", new_cost)
        print("delta J", delta_J)
        z = (old_cost - new_cost) / -delta_J

        if z < self.c1:
            restart = True
            self.alpha = self.alpha - 1e-2 if self.alpha > 1e-4 else 1e-4

            print("z = ", z, ", alpha = ", self.alpha)
            return None, None, restart

        return x_seq_hat, u_seq_hat, False
