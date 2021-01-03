import autograd.numpy as np

from autograd import jacobian, grad
from tqdm import trange

import logging

from numpy.linalg import LinAlgError


def boxQP(H, g, lower, upper, x0):
    n = H.shape[0]
    clamped = np.zeros(n)
    free = np.ones(n)
    Hfree = np.zeros(n)
    oldvalue = 0
    result = 0
    nfactor = 0
    clamp = lambda value: np.maximum(lower, np.minimum(upper, value))

    maxIter = 100
    minRelImprove = 1e-8
    minGrad = 1e-8
    stepDec = 0.6
    minStep = 1e-22
    Armijo = 0.1

    if x0.shape[0] == n:
        x = clamp(x0)
    else:
        lu = np.array([lower, upper])
        lu[np.isnan(lu)] = np.nan
        x = np.nanmean(lu, axis=1)

    value = np.dot(x.T, np.dot(H, x)) + np.dot(x.T, g)

    for iteration in range(maxIter):
        if result != 0:
            break

        if iteration > 1 and (oldvalue - value) < minRelImprove * abs(oldvalue):
            result = 4
            logging.info("[QP info] Improvement smaller than tolerance")
            break

        oldvalue = value

        grad = g + np.dot(H, x)

        old_clamped = clamped
        clamped = np.zeros(n)
        clamped[np.logical_and(x == lower, grad > 0)] = 1
        clamped[np.logical_and(x == upper, grad < 0)] = 1
        free = np.logical_not(clamped)

        if np.all(clamped):
            result = 6
            logging.info("[QP info] All dimensions are clamped")
            break

        if iteration == 0:
            factorize = True
        else:
            factorize = np.any(old_clamped != clamped)

        if factorize:
            try:
                if not np.all(np.allclose(H, H.T)):
                    H = np.triu(H)
                Hfree = np.linalg.cholesky(H[np.ix_(free, free)])
            except LinAlgError:
                eigs, _ = np.linalg.eig(H[np.ix_(free, free)])
                print(eigs)
                result = -1
                logging.info("[QP info] Hessian is not positive definite")
                break
            nfactor += 1

        gnorm = np.linalg.norm(grad[free])
        if gnorm < minGrad:
            result = 5
            logging.info("[QP info] Gradient norm smaller than tolerance")
            break

        grad_clamped = g + np.dot(H, x*clamped)
        search = np.zeros(n)

        y = np.linalg.lstsq(Hfree.T, grad_clamped[free])[0]
        search[free] = -np.linalg.lstsq(Hfree, y)[0] - x[free]
        sdotg = np.sum(search*grad)
        if sdotg >= 0:
            print(f"[QP info] No descent direction found. Should not happen. Grad is {grad}")
            break

        # armijo linesearch
        step = 1
        nstep = 0
        xc = clamp(x + step*search)
        vc = np.dot(xc.T, g) + 0.5*np.dot(xc.T, np.dot(H, xc))
        while (vc - oldvalue) / (step*sdotg) < Armijo:
            step *= stepDec
            nstep += 1
            xc = clamp(x + step * search)
            vc = np.dot(xc.T, g) + 0.5 * np.dot(xc.T, np.dot(H, xc))

            if step < minStep:
                result = 2
                break

        # accept candidate
        x = xc
        value = vc

        # print(f"[QP info] Iteration {iteration}, value of the cost: {vc}")

    if iteration >= maxIter:
        result = 1

    return x, result, Hfree, free


class DDP_v1:
    """
    Heavily based on this implementation: https://github.com/neka-nat/ilqr-gym
    This variant of DDP is from papers: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf and
                                        https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
    """
    def __init__(self,
                 coeff,
                 dynamics_model,
                 umax,
                 state_dim,
                 action_dim,
                 forward_dynamics=None,
                 reg_type=0,
                 pred_time=50
                 ):
        self.reg_type = reg_type
        self.apply_control_constrains = True
        self.control_limits = np.array([[-umax, umax]])  # ,[-1, 1]])

        self.alpha = 10**np.linspace(0, -4, 11)

        self.a1, self.a2, self.a3 = coeff
        self.forward_dynamics = forward_dynamics

        self.pred_time = pred_time
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.umax = umax

        self.lf = self.final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)

        self.l = lambda x, u: 0.5*np.sum(np.square(u), axis=0 if len(u.shape) == 1 else 1)
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

    def final_cost(self, state):
        '''
        if len(state.shape) >= 2:
            goal = np.repeat(self.goal[np.newaxis, :], repeats=state.shape[0], axis=0)
            achieved_goal = np.transpose(self.forward_dynamics(state[:, 0], state[:, 2]), [1, 0])
            cost = self.a1 * np.sum(np.square(goal - achieved_goal), axis=1) + self.a2 * np.square(state[:, 1]) + \
                                                                    self.a3 * np.square(state[:, 3])
        else:
            cost = self.a1 * np.sum(np.square(self.goal - self.forward_dynamics(state[0], state[2]))) + \
                   self.a2 * np.square(state[1]) + self.a3 * np.square(state[3])
        '''
        if len(state.shape) >= 2:
            cost = 0.5 * (np.square(1.0 - np.cos(state[:, 2])) + np.square(state[:, 1]) + np.square(state[:, 3]))

        else:
            cost = 0.5 * (np.square(1.0 - np.cos(state[2])) + np.square(state[1]) + np.square(state[3]))

        return cost

    def backward(self, x, u, iter_number, lambda_):
        print("[INFO] Start backward pass, iteration:", iter_number)

        v = np.array([0.0, 0.0])
        v_x = np.zeros((self.pred_time + 1, self.state_dim))
        v_xx = np.zeros((self.pred_time + 1, self.state_dim, self.state_dim))

        # self.v[-1] += self.lf(x_seq[-1])
        v_x[-1] = self.lf_x(x[-1])
        v_xx[-1] = self.lf_xx(x[-1])

        k = np.zeros((self.pred_time, self.action_dim))
        K = np.zeros((self.pred_time, self.action_dim, self.state_dim))

        diverged_iteration = 0
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x[t], u[t])
            f_u_t = self.f_u(x[t], u[t])
            q_x = self.l_x(x[t], u[t]) + np.dot(f_x_t.T, v_x[t + 1])
            q_u = self.l_u(x[t], u[t]) + np.dot(f_u_t.T, v_x[t + 1])
            q_xx = self.l_xx(x[t], u[t]) + np.dot(np.dot(f_x_t.T, v_xx[t + 1]), f_x_t) + \
                    np.einsum("i,ijk->jk", v_x[t + 1], self.f_xx(x[t], u[t]))
            q_ux = self.l_ux(x[t], u[t]) + np.dot(np.dot(f_u_t.T, v_xx[t + 1]), f_x_t) + \
                    np.einsum('i,ijk->jk', v_x[t + 1], self.f_ux(x[t], u[t]))
            q_uu = self.l_uu(x[t], u[t]) + np.dot(np.dot(f_u_t.T, v_xx[t + 1]), f_u_t) + \
                    np.einsum('i,ijk->jk', v_x[t + 1], self.f_uu(x[t], u[t]))

            v_xx_reg = v_xx[t + 1] + (self.reg_type == 2)*lambda_*np.eye(self.state_dim)
            q_ux_reg = self.l_ux(x[t], u[t]) + np.dot(np.dot(f_u_t.T, v_xx_reg), f_x_t) + \
                   np.einsum('i,ijk->jk', v_x[t + 1], self.f_ux(x[t], u[t]))

            q_uu_F = self.l_uu(x[t], u[t]) + np.dot(np.dot(f_u_t.T, v_xx_reg), f_u_t) + \
                     np.einsum('i,ijk->jk', v_x[t + 1], self.f_uu(x[t], u[t])) + \
                     (self.reg_type == 1)*lambda_*np.eye(self.action_dim)
            #print("l_uu = ", self.l_uu(x_seq[t], u_seq[t]))
            #print("f_u.T V_xx f_u", np.dot(np.dot(f_u_t.T, v_xx_reg), f_u_t))
            #print("V_x f_uu", np.einsum('i,ijk->jk', self.v_x[t + 1], self.f_uu(x_seq[t], u_seq[t])))
            #print("lambda I", (self.reg_type == 1)*lambda_*np.eye(self.action_dim))
            #print("q_uu_F", q_uu_F)

            if not np.all(np.allclose(q_uu_F, q_uu_F.T)):
                q_uu_F = np.triu(q_uu_F)

            if not self.apply_control_constrains:
                try:
                    L = np.linalg.cholesky(q_uu_F)
                except LinAlgError as e:
                    logging.warning(f"Q_uu_F is not positive-definite, Q_uu_F is {q_uu_F}, "
                                    "Q_uu_F eigenvalues are {np.linalg.eigvals(q_uu_F))}")
                    diverged_iteration = t
                    break

                kK = -np.linalg.lstsq(L, np.linalg.lstsq(L.T, np.concatenate([q_u[:, np.newaxis], q_ux_reg], axis=1))[0])[0]

                k_i = -kK[:, 0]
                K_i = -kK[:, 1:]
                assert k_i.shape == (self.action_dim,), f"k.shape is, {k_i.shape}"
                assert K_i.shape == (self.action_dim, self.state_dim), f"K.shape is, {K_i.shape}"
            else:
                lower = self.control_limits[:, 0] - u[t]
                upper = self.control_limits[:, 1] - u[t]
                k_i, result, L, free = boxQP(q_uu_F, q_u, lower, upper, x0=k[min(t+1, self.pred_time - 1), :])

                #print("Norm of k_i", np.linalg.norm(k_i), "Norm of K_i:", np.linalg.norm(K))
                if result < 1:
                    print(f"[INFO] Backward pass was diverged at iteration {t}! Lambda is:", lambda_)
                    diverged_iteration = t
                    break

                K_i = np.zeros((self.action_dim, self.state_dim))
                if np.any(free):
                    Lfree = -np.linalg.lstsq(L, np.linalg.lstsq(L.T, q_ux_reg[free, :])[0])[0]
                    K_i[free, :] = Lfree

            dv = np.array([np.dot(k_i.T, q_u), 0.5*np.dot(k_i.T, np.dot(q_uu, k_i))])
            v += dv
            v_x[t] = q_x + np.dot(K_i.T, np.dot(q_uu, k_i)) + np.dot(K_i.T, q_u) + np.dot(q_ux.T, k_i)
            v_xx[t] = q_xx + np.dot(K_i.T, np.dot(q_uu, K_i)) + np.dot(K_i.T, q_ux) + np.dot(q_ux.T, K_i)
            # v_xx[t] = 0.5*(v_xx[t] + v_xx[t].T)

            k[t, :] = k_i
            K[t, :, :] = K_i

        return diverged_iteration, k, K, v

    def get_cost(self, states, controls):
        J = 0
        for t in range(self.pred_time - 1):
            running_cost = self.l(states[t], controls[t])
            J += running_cost

        print("[Cost INFO] Running cost:", J, end=',')

        final_cost = self.lf(states[-1])
        print(" Final cost:", final_cost, end=',')
        J += final_cost
        print("Total cost:", J)

        return J

    def forward(self, x, u, k, K, iter_number):
        print("[INFO] Start forward pass, iteration number:", iter_number)
        n_alpha = self.alpha.shape[0]

        x_new = np.zeros((n_alpha, self.pred_time + 1, self.state_dim))
        x_new[:, 0, :] = np.repeat(x[0, :][np.newaxis, :], repeats=n_alpha, axis=0)
        u_new = np.zeros((n_alpha, self.pred_time, self.action_dim))
        c_new = np.zeros((n_alpha, self.pred_time + 1))

        for t in range(self.pred_time):
            u_new[:, t, :] = np.repeat(u[t, :][np.newaxis, :], repeats=n_alpha, axis=0) + np.outer(self.alpha, k[t]) + \
             np.einsum("mn,an->am", K[t, :, :], x_new[:, t, :] - np.repeat(x[t, :][np.newaxis, :], repeats=n_alpha, axis=0))

            u_new[:, t, :] = np.maximum(np.minimum(u_new[:, t, :], self.umax), -self.umax)
            x_new[:, t+1, :] = self.f(x_new[:, t, :], u_new[:, t, :])
            c_new[:, t] = self.l(x_new[:, t, :], u_new[:, t, :])

        c_new[:, self.pred_time] = self.lf(x_new[:, self.pred_time, :])

        return x_new, u_new, c_new
