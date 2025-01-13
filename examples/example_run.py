import numpy as np
import matplotlib.pyplot as plt
from src.forward_solver import solve_forward
from src.optimize import optimize_boundary


# Parameters
alpha = 0.01  # thermal diffusivity
L = 1.0  # length of the rod in meters
Nx = 50  # number of spatial points
Nt = 100  # number of time steps
T = 10.0  # total time in seconds
dx = L / Nx
dt = (dx**2) / (4 * alpha)  # Adjust dt based on the CFL condition


# Initial and target temperature distributions
u0 = np.zeros(Nx)
u_target = np.linspace(0, 100, Nx)

# Optimization
optimal_boundary_temp = optimize_boundary(u0, alpha, dx, dt, Nx, Nt, L, u_target)

# Solve the forward problem with optimized boundary control
def boundary_temp(t):
    return optimal_boundary_temp  # constant boundary temperature

u_sol = solve_forward(u0, boundary_temp, alpha, dx, dt, Nx, Nt)

# Visualization
plt.plot(np.linspace(0, L, Nx), u_sol[-1], label='Optimized Temperature')
plt.plot(np.linspace(0, L, Nx), u_target, label='Target Temperature', linestyle='--')
plt.xlabel('Position along the rod')
plt.ylabel('Temperature')
plt.title('1D heat equation boundary control optimization results')
plt.legend()
plt.show()