import gym

from robotics.OptimalControl.DifferentialDynamicProgramming.DDP import DDP_v1

import numpy as np
import autograd.numpy as anp
from matplotlib import pyplot as plt


def main():

    env = gym.make('CartPoleContinuous-v0').env
    T = 200
    print("Horizon", T)

    run_name = "mpc"

    ddp = DDP_v1(coeff=(100, 1, 1),
                 dynamics_model=env._state_eq,
                 # forward_dynamics=env.forward_dynamics,
                 umax=1,
                 state_dim=env.observation_space.shape[0],  # ["observation"]
                 action_dim=env.action_space.shape[0],
                 pred_time=500,
                 reg_type=2
                 )

    def execute_actions(initial_state, actions_sequence):
        states_sequence = [initial_state]
        state = initial_state
        for t in range(ddp.pred_time):
            next_state = env._state_eq(state, actions_sequence[t])
            states_sequence.append(next_state)
            state = next_state

        return np.array(states_sequence)

    state = env.reset()
    # goal = np.array([1, 1, 0.08])
    # initial_distance = np.linalg.norm(state["achieved_goal"] - goal)

    # ddp.set_goal(goal[:2])
    actions = np.zeros((T, env.action_space.shape[0]))

    u = np.array([1e-2*anp.random.random(size=1) for _ in range(ddp.pred_time)])
    x = execute_actions(state, u)  # ['observation']
    assert np.all(np.abs(x) < 1e8), "States diverged!"
    cost = ddp.get_cost(x, u)

    H = 50
    assert H <= ddp.pred_time

    # DDP stuff
    lambda_ = 1
    dlambda = 1
    lambda_factor = 1.2
    min_lambda = 1e-6
    max_lambda = 1e4
    zMin = 0

    # DDP iterations
    for iteration in range(0, T, H):
        for epoch in range(50):
            backPassDone = False
            while not backPassDone:
                diverged_iteration, k, K, v = ddp.backward(x, u, epoch, lambda_)

                if diverged_iteration:
                    print(f"[INFO] Increase lambda, because backward pass diverged! Now lambda is {lambda_}")
                    dlambda = max(dlambda*lambda_factor, lambda_factor)
                    lambda_ = max(lambda_*dlambda, min_lambda)
                    if lambda_ > max_lambda:
                        print("[INFO] Lambda exceeds max value")
                        return
                    continue

                backPassDone = True

            # check for termination due to a small gradient
            g_norm = np.mean(np.max(np.abs(k) / (np.abs(u) + 1), axis=0))
            if g_norm < 1e-4 and lambda_ < 1e-5:
                lambda_ = min(dlambda / lambda_factor, 1 / lambda_factor)
                lambda_ = lambda_*dlambda*(lambda_ > min_lambda)
                print("Success! gradient norm < 1e-4")
                break

            forwardPassDone = False
            if backPassDone:
                x_new, u_new, cost_new = ddp.forward(x, u, k, K, epoch)
                Dcost = np.sum(cost) - np.sum(cost_new, axis=1)

                dcost, w = np.max(Dcost), np.argmax(Dcost)
                alpha = ddp.alpha[w]
                expected = -alpha*(v[0] + alpha*v[1])

                if expected > 0:
                    z = dcost / expected
                else:
                    z = np.sign(dcost)
                    print('non-positive expected reduction: should not occur')

                if z > zMin:
                    forwardPassDone = True
                    cost_new = cost_new[w, :]
                    x_new = x_new[w, :, :]
                    u_new = u_new[w, :, :]

            if forwardPassDone:
                # decrease lambda
                print(f"[INFO] Decrease lambda! Now lambda is {lambda_}")
                dlambda = min(dlambda / lambda_factor, 1 / lambda_factor)
                lambda_ = lambda_ * dlambda * (lambda_ > min_lambda)

                # accept changes
                u = u_new
                x = x_new
                cost = cost_new
                print("[INFO] Iteration was done, cost is", np.sum(cost))

                if dcost < 1e-7:
                    print(f"[INFO] Success!!! Lambda is: {lambda_}, cost is: {np.sum(cost)}")
                    break
            else:  # no cost improvement
                print(f"[INFO] Cost wasn't improved, old cost is: {np.sum(cost)}, new cost is: {np.sum(cost_new)}")
                # increase lambda
                print(f"[INFO] Increase lambda, because forward pass wasn't successful! Now lambda is {lambda_}")
                dlambda = max(dlambda * lambda_factor, lambda_factor)
                lambda_ = max(lambda_ * dlambda, min_lambda)

                if lambda_ > max_lambda:
                    print("[INFO] Lambda exceeds its max value")
                    raise StopIteration("Lambda exceeds its max value")

        print(iteration, "->", iteration+H)
        actions[iteration:iteration+H, :] = u[:H, :]

        for inner_time in range(H):
            next_state, reward, done, _ = env.step(u[inner_time, :])

        # distance = np.linalg.norm(next_state['achieved_goal'] - goal)
        # expected_distance = np.linalg.norm(env.forward_dynamics(x[-1, 0], x[-1, 2]) - goal[:2])
        # print(f"Distance to the goal after {H} moves: {distance}, expected distance to the goal after {ddp.pred_time}"
        #      f"moves: {expected_distance}, Initial distance: {initial_distance}")

        if iteration >= T - 1:  # distance < env.distance_threshold:
            break

        u = np.array([a for a in u[H:]] + [np.zeros(1) for _ in range(H)])
        x = execute_actions(next_state, u)  # ['observation']

    actions = np.array(actions + [np.zeros(1) for _ in range(T - len(actions))])
    print("Action length:", actions.shape)
    np.save(f"{run_name}_actions", actions)

    # testing
    print("\n[INFO] Testing")
    state = env.reset()
    # env.set_goal(goal)
    distances = []
    for t in range(T):
        next_state, reward, done, _ = env.step(actions[t])
        # distance = np.linalg.norm(next_state['achieved_goal'] - goal)
        # distances.append(distance)
        # print(f"Distance to the goal: {distance}")
        env.render()
        if done:
            break

    plt.plot(distances)
    plt.show()


def boxQP_test_():
    from robotics.OptimalControl.DifferentialDynamicProgramming.DDP import boxQP

    g = np.array([1, 1])  # np.random.randn(n)
    H = np.array([[2, 0], [0, 2]])  # np.random.randn(n, n)
    lower = -np.ones(2)
    upper = np.ones(2)
    x_0 = np.array([1000, 1000])

    x, result, Hfree, free = boxQP(H, g, lower, upper, x_0)
    print("result is", result)
    print("x = ", x)
    # print(Hfree)


if __name__ == '__main__':
    main()
