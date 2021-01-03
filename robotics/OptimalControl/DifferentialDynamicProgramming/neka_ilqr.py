import gym
import Reach_v0
from autograd import grad, jacobian
import autograd.numpy as np

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

class ILqr:
    def __init__(self, next_state, running_cost, final_cost,
                 umax, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.umax = umax

        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def backward(self, x_seq, u_seq):
        v = [0.0 for _ in range(self.pred_time + 1)]
        v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
                   np.einsum("i,ijk->jk", self.v_x[t + 1], self.f_xx(x_seq[t], u_seq[t]))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(np.matmul(f_u_t.T, self.v_xx[t + 1]), f_x_t) + \
                   np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_ux(x_seq[t], u_seq[t]))
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(np.matmul(f_u_t.T, self.v_xx[t + 1]), f_u_t) + \
                   np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_uu(x_seq[t], u_seq[t]))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])

        return x_seq_hat, u_seq_hat


env = gym.make('Reach-v3')
obs = env.reset()
goal = obs['desired_goal']

ilqr = ILqr(lambda x, u: env.get_next_state(x, u),  # x(i+1) = f(x(i), u)
            lambda x, u: 0.5 * np.sum(np.square(u)),  # l(x, u)
            lambda state: 10 * np.sum(np.square(goal[:2] - env.forward_dynamics(state[0], state[2]))) +
                          np.square(state[1]) + np.square(state[3]),  # lf(x)
            env.max_u,
            env.observation_space['observation'].shape[0],
            pred_time=50)

u_seq = [1e-2*np.random.randn(2) for _ in range(ilqr.pred_time)]
x_seq = [obs['observation']]
for t in range(ilqr.pred_time):
    x_seq.append(env.get_next_state(x_seq[-1], u_seq[t]))

from tqdm import trange
actions = []
while True:
    # env.render(mode="human")
    for _ in trange(50):
        k_seq, kk_seq = ilqr.backward(x_seq, u_seq)
        x_seq, u_seq = ilqr.forward(x_seq, u_seq, k_seq, kk_seq)

    print("action", u_seq[0])
    for i in range(10):
        obs, _, _, _ = env.step(u_seq[i])

    distance = np.linalg.norm(obs['achieved_goal'] - goal)
    print("Distance to the goal is", distance, "Expected distance is",
          np.linalg.norm(env.forward_dynamics(x_seq[-1][0], x_seq[-1][2]) - goal[:2]))

    x_seq[0] = obs['observation']
    u_seq = np.array([a for a in u_seq[10:]] + [1e-2*np.random.randn(2) for _ in range(10)])

    for i in range(1, ilqr.pred_time):
        x_seq[i] = env.get_next_state(x_seq[i-1], u_seq[i-1])
