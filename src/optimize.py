from src.forward_solver import solve_forward
from src.adjoint_solver import solve_adjoint
import numpy as np


def compute_cost(u_sol, u_target, dx, boundary_temp=None, penalty=0.0):
    # Calculate the cost function value
    cost_data = np.sum((u_sol[-1] - u_target)**2) * dx

    # Add a penalty value at the boundary
    cost_reg = 0.0
    if (penalty > 0.0) and (boundary_temp is not None):
        cost_reg = penalty * (boundary_temp**2)
    return cost_data + cost_reg

def compute_gradient(lambda_, dt):
    gradient = np.sum(lambda_[:, 0]) * dt
    return gradient

def optimize_boundary(u0, alpha, dx, dt, Nx, Nt, u_target, max_iter=20, learning_rate=1e-4,
                       boundary_min=0.0, boundary_max=500.0,penalty=0.0):
    # Initialize boundary control guess
    boundary_control_value = np.random.uniform(0, 100)  # Single value control at the boundary
    boundary_control = lambda t: boundary_control_value  # Convert to a callable function

    for k in range(max_iter):

        # Apply boundary control and solve forward problem
        u_sol = solve_forward(u0, boundary_control, alpha, dx, dt, Nx, Nt) 
        J_old = compute_cost(u_sol, u_target, dx, boundary_temp=boundary_control_value, penalty=penalty)

        # Solve adjoint problem
        lambda_ = solve_adjoint(u_sol, u_target, alpha, dx, dt, Nx, Nt)
        grad = compute_gradient(lambda_, dt)

        # Line search
        # Try several possible step sizes and find the optimal one
        step_candidates = [learning_rate, learning_rate*0.5, learning_rate*0.1, learning_rate*0.01, 1e-6]
        updated = False
        for step in step_candidates:
            candidate_value = boundary_control_value - step * grad
            candidate_value = np.clip(candidate_value, boundary_min, boundary_max) # In case the temperature out of bound
            
            # Use forward function to check whether the cost decrease
            candidate_control = lambda t: candidate_value
            u_sol_test = solve_forward(u0, candidate_control, alpha, dx, dt, Nx, Nt)
            J_new = compute_cost(u_sol_test, u_target, dx, boundary_temp=candidate_value, penalty=penalty)

            if J_new < J_old:
                boundary_control_value = candidate_value
                boundary_control = candidate_control
                updated = True
                print(f'[Iter {k+1}] accept step={step:e}, J from {J_old:.4e} to {J_new:.4e}')
                break
        # Remain same if all stepsizes fail
        if not updated:
            print(f'[Iter {k+1}] no improvement, cost={J_old:.4e}')


    return boundary_control_value