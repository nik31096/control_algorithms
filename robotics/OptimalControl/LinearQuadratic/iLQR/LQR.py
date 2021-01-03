import numpy as np
from matplotlib import pyplot as plt


class LQR_TI:
    """
    Time-invariant LQR version for discrete-time systems with dynamics equation: x_{t+1} = A x_t + B u_t
    """
    def __init__(self, T):
        self.T = T
        self.p = [0]*self.T

    def set_parameters(self, A, B, Q, R, x_T, Q_f=None):
        self.A, self.B, self.Q, self.R, self.x_T = A, B, Q, R, x_T
        if Q_f is None:
            self.Q_f = Q
        else:
            self.Q_f = Q_f

    def get_gain_sequence(self):
        p = self.Q_f
        self.p[0] = p
        for t in range(1, self.T):
            inv_part = np.linalg.inv(np.identity(self.x_T.shape[0]) +
                                     np.dot(p, np.dot(self.B, np.dot(np.linalg.inv(self.R), self.B.T))))
            p = np.dot(self.A.T, np.dot(inv_part, np.dot(p, self.A))) + self.Q
            self.p[t] = p

    def get_control_gain(self, t):
        p_next = self.p[t+1]
        k = -np.dot(np.linalg.inv(self.R + np.dot(self.B.T, np.dot(p_next, self.B))),
                    np.dot(self.B.T, np.dot(p_next, self.A)))

        return k

    def get_control(self, x, t):
        k = self.get_control_gain(t)
        return np.dot(k, x - self.x_T), k


class LQR_TV(LQR_TI):
    """
    Time-varying LQR version
    """
    def __init__(self, T):
        super(LQR_TV, self).__init__(T)

    def get_k_gains(self):
        k_gains = []
        p_t_1 = self.Q_f
        for t in range(self.T):
            try:
                inv_part = np.linalg.inv(np.eye(self.x_T.shape[0]) +
                           np.dot(p_t_1, np.dot(self.B[t], np.dot(np.linalg.inv(self.R[t]), self.B[t].T))))
            except np.linalg.LinAlgError as e:
                print(self.R[t], np.linalg.eigvals(self.R[t]))
                raise e
            p_t = np.dot(self.A[t].T, np.dot(inv_part, np.dot(p_t_1, self.A[t]))) + self.Q[t]
            k = np.dot(np.linalg.inv(self.R[t] + np.dot(self.B[t].T, np.dot(p_t_1, self.B[t]))),
                       np.dot(self.B[t].T, np.dot(p_t_1, self.A[t])))
            k_gains.append(k)
            p_t_1 = p_t

        return k_gains[::-1]


if __name__ == '__main__':
    # time-invariant linear system control example using LQR
    x_0 = np.array([5, 3])
    x_T = np.array([2, 4])
    A = np.array([[1, 0],
                  [2, -1]])
    B = np.array([[1, -2, 0],
                  [2, 0, 1]])
    Q = np.array([[1, 0],
                  [0, 1]])
    R = np.array([[10, 0, 0],
                  [0, 10, 0],
                  [0, 0, 10]])
    Q_f = np.array([[100, 0],
                    [0, 100]])

    lqr = LQR_TI(T=10)
    lqr.set_parameters(A, B, Q, R, x_T, Q_f=Q_f)
    lqr.get_gain_sequence()

    dyn_shifted = lambda x, u: np.dot(A, x - x_T) + np.dot(B, u) + x_T

    states, states2, actions, actions2 = [], [], [], []
    shifted_states = []
    x = x_0
    states.append(x_0)
    for t in range(lqr.T - 1):
        u, k = lqr.get_control(x, t)
        print("System eigenvalues:", np.linalg.eigvals(A + np.dot(B, k)))
        x = dyn_shifted(x, u)
        states.append(x)
        actions.append(u)

    print("Final state is:", x, ", final control is:", u)

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot([x[0] for x in states], label="Real states")
    plt.ylim((-4, 6))
    plt.legend()
    plt.title("First state component")
    plt.subplot(122)
    plt.plot([x[1] for x in states], label="Real states")
    plt.ylim((-4, 6))
    plt.legend()
    plt.title("Second state component")
    plt.show()
