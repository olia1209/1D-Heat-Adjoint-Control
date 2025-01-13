import numpy as np
import matplotlib.pyplot as plt
from src.forward_solver import solve_forward
from src.optimize import optimize_boundary


# Parameters
alpha = 0.01  # thermal diffusivity
L = 1.0  # length of the rod in meters
Nx = 50  # number of spatial points
T = 10.0  # total time in seconds
dx = L / Nx
dt = (dx**2) / (4 * alpha)  # Adjust dt based on the CFL condition
Nt = int(T / dt)  # number of time steps


# Initial and target temperature distributions
u0 = np.zeros(Nx)
x = np.linspace(0, 1, Nx)
#u_target = 80*(x - 1)**2 + 80 # Parabola
#u_target = np.linspace(200, 80, Nx) # Linear
u_target = 100 - 40 * np.sin(2 * np.pi * x) # Sine


# Optimization
optimal_boundary_temp = optimize_boundary(u0, alpha, dx, dt, Nx, Nt, u_target, max_iter=20, learning_rate=1e-4,
                       boundary_min=0.0, boundary_max=200.0,penalty=0.0)

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