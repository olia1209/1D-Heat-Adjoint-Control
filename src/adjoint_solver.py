import numpy as np


def solve_adjoint(u_sol, u_target, alpha, dx, dt, Nx, Nt):
    lambda_ = np.zeros((Nt, Nx))
    lambda_[-1, :] = u_sol[-1, :] - u_target
    for n in range(Nt - 1, 0, -1):
        for i in range(1, Nx - 1):
            lambda_[n-1, i] = lambda_[n, i] - alpha * dt / dx**2 * (lambda_[n, i+1] - 2*lambda_[n, i] + lambda_[n, i-1])
        # Apply boundary condition
        lambda_[n-1, 0] = lambda_[n-1, 1]
        lambda_[n-1, -1] = lambda_[n-1, -2]
    return lambda_
