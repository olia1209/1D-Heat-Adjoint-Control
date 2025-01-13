from src.forward_solver import solve_forward
from src.adjoint_solver import solve_adjoint
import numpy as np


def compute_cost(u_sol, u_target, dx):
    # Calculate the cost function value
    return np.sum((u_sol - u_target)**2) * dx

def compute_gradient(lambda_, dt):
    gradient = np.sum(lambda_[:, 0]) * dt
    return gradient

def optimize_boundary(u0, alpha, dx, dt, Nx, Nt, L, u_target, max_iter=10):
    # Initialize boundary control guess
    boundary_control_value = np.random.rand()  # Single value control at the boundary
    boundary_control = lambda t: boundary_control_value  # Convert to a callable function

    for k in range(max_iter):
        # Apply boundary control and solve forward problem
        u_sol = solve_forward(u0, boundary_control, alpha, dx, dt, Nx, Nt) # 把temp改成了control
        J = compute_cost(u_sol, u_target, dx)

        # Solve adjoint problem
        lambda_ = solve_adjoint(u_sol, u_target, alpha, dx, dt, Nx, Nt)
        grad = compute_gradient(lambda_, dt)
        
        eta = 0.01  # Learning rate
        boundary_control_value -= eta * grad
        boundary_control = lambda t: boundary_control_value  # Update the function
    
        print(f'Iteration {k+1}, Cost: {J}')

    return boundary_control_value