import numpy as np
import matplotlib.pyplot as plt
from src.forward_solver import solve_forward

# ----------Example for forward solver----------------------------------
# Define parameters
L = 1.0       # Length of the rod
T = 0.1       # Total time
dx = 0.01     # Space step size
dt = 0.0001   # Time step size
alpha = 0.01  # Thermal diffusivity

# Initial temperature distribution
u0 = np.zeros(int(L / dx) + 1)

# Boundary temperature control (e.g., constant value at left boundary)
def boundary_temp(t):
    return 100.0  # Keep left boundary at 100 degrees

# Solve the heat equation
u = solve_forward(u0, boundary_temp, alpha, dx, dt, T, L)

# Plot the result at the final time step
x = np.linspace(0, L, int(L / dx) + 1)
plt.plot(x, u[-1, :], label="t = {:.2f}".format(T))
plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("Temperature Distribution at Final Time")
plt.legend()
plt.grid(True)
plt.show()
