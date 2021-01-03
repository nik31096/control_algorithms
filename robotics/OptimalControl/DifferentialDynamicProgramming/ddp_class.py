import autograd.numpy as np
from numpy.linalg import LinAlgError
from autograd import grad, jacobian

from tqdm import trange


class DDP_v2:
    """
    Heavily based on https://github.com/neka-nat/ilqr-gym and https://github.com/studywolf/control
    """
    def __init__(self, a1, a2, a3, dynamics_model, forward_dynamics, umax, state_dim, pred_time=50):
        self.lamb = 1
        self.lamb_factor = 10
        self.forward_dynamics = forward_dynamics

        self.pred_time = pred_time
        self.state_dim = state_dim
        self.umax = umax
        self.goal = np.zeros(2)
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.lf = lambda x: 0.5*np.sum(a1*(self.goal - forward_dynamics(x[0], x[2]))**2 +
                                       a2*np.square(x[1]) + a3*np.square(x[3]))
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)

        self.l = lambda x, u: 0.5*np.sum(np.square(u))
        self.l_x = grad(self.l, 1)
        self.l_u = grad(self.l, 2)
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

    def get_cost(self, initial_state, controls):
        J = self.l(initial_state, controls[0])
        state = initial_state
        for i in range(len(controls) - 2):
            next_state = self.f(state, controls[i])
            running_cost = self.l(next_state, controls[i+1])
            J += running_cost
            state = next_state

        print("Running cost:", J, end=',')

        final_cost = self.lf(state)
        print(" Final cost:", final_cost, end=', ')
        J += final_cost

        print("Total cost:", J)

        return J

    def plan(self, x_seq, u_seq, epoch, iterations):
        for iteration in range(iterations):
            print(f"\n[INFO] Start backward pass, epoch: {epoch}, iteration: {iteration}")
            self.v[-1] = self.lf(x_seq[-1])
            self.v_x[-1] = self.lf_x(x_seq[-1])
            self.v_xx[-1] = self.lf_xx(x_seq[-1])
            print(f"Initial conditions: v[N] = {self.v[-1]}, v_x[N] = {self.v_x[-1]}, v_xx[N] = {self.v_xx[-1]}")
            k_seq = []
            K_seq = []
            #print(len(x_seq), self.pred_time)
            for t in range(self.pred_time - 2, -1, -1):
                f_x_t = self.f_x(x_seq[t], u_seq[t])
                f_u_t = self.f_u(x_seq[t], u_seq[t])
                q_x = self.l_x(x_seq[t], u_seq[t], ) + np.dot(f_x_t.T, self.v_x[t + 1])
                q_u = self.l_u(x_seq[t], u_seq[t]) + np.dot(f_u_t.T, self.v_x[t + 1])
                q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
                  np.dot(np.dot(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
                  np.einsum("i,ijk->jk", self.v_x[t + 1], self.f_xx(x_seq[t], u_seq[t]))

                tmp = np.dot(f_u_t.T, self.v_xx[t + 1])
                q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.dot(tmp, f_u_t) + \
                       np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_uu(x_seq[t], u_seq[t]))

                try:
                    eig_values, eig_vectors = np.linalg.eig(q_uu)
                except LinAlgError as e:
                    print("f_u = ", f_u_t)
                    print("v_x = ", self.v_x[t+1])
                    print("v_xx' = ", self.v_xx[t+1])
                    print("l_uu = ", self.l_uu(x_seq[t], u_seq[t]))

                    raise e

                eig_values[eig_values < 0] = 0.0
                eig_values += self.lamb

                q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.dot(tmp, f_x_t) + \
                  np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))

                inv_q_uu = np.dot(eig_vectors, np.dot(np.diag(1.0 / eig_values), eig_vectors.T))
                k = -np.dot(inv_q_uu, q_u)
                K = -np.dot(inv_q_uu, q_ux)
                dv = -0.5*np.dot(k.T, np.dot(q_uu, k))
                self.v[t] += dv
                self.v_x[t] = q_x - np.dot(K.T, np.dot(q_uu, k))
                self.v_xx[t] = q_xx - np.dot(K.T, np.dot(q_uu, K))
                k_seq.append(k)
                K_seq.append(K)

            k = k_seq[::-1]
            K = K_seq[::-1]

            print(f"[INFO] Start forward pass, epoch: {epoch}, iteration: {iteration}")
            x_seq_hat = np.array(x_seq)
            u_seq_hat = np.array(u_seq)
            for t in range(len(u_seq)):
                control = k[t] + np.matmul(K[t], x_seq_hat[t] - x_seq[t])
                new_control = u_seq[t] + control
                u_seq_hat[t] = np.clip(new_control, -self.umax, self.umax)
                x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])

            print("Expected distance to the goal is",
                  np.linalg.norm(self.forward_dynamics(x_seq_hat[-1][0], x_seq_hat[-1][2]) - self.goal),
                  ", Expected end-effector position:", self.forward_dynamics(x_seq_hat[-1][0], x_seq_hat[-1][2]),
                  ", Goal position:", self.goal)

            print("Old cost:", end=' ')
            old_cost = self.get_cost(x_seq[0], u_seq)
            print("New cost:", end=' ')
            new_cost = self.get_cost(x_seq[0], u_seq_hat)

            if new_cost < old_cost:
                x_seq = x_seq_hat.copy()
                u_seq = u_seq_hat.copy()
                restart = False
                self.lamb /= self.lamb_factor
            else:
                restart = True
                print(f"[INFO] Iteration {iteration} should be repeated")
                # self.lamb_factor is > 1
                self.lamb = max(self.lamb * self.lamb_factor, 1e5)
                break

        return u_seq_hat, restart
