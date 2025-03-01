import numpy as np

def solve_forward(u0, boundary_temp, alpha, dx, dt, Nx, Nt):
    """
    Solve the 1D heat equation using explicit finite difference method.
    
    Parameters:
        u0: ndarray
            Initial temperature distribution along the x-axis.
        boundary_temp: function
            Function for boundary control at x=0 (left boundary). Should take time 't' as argument and return a float.
        alpha: float
            Thermal diffusivity constant.
        dx: float
            Space step size.
        dt: float
            Time step size.
        Nx: integer
            Number of spatial points
        Nt: integer
            Number of time steps
        
    
    Returns:
        u: ndarray
            Numerical solution of u(t, x).
    """
    
    # Initialize solution array
    u = np.zeros((Nt+1, Nx))
    u[0, :] = u0  # Set initial condition
    
    # Time-stepping loop
    for n in range(Nt):
        for i in range(1, Nx-1):
            u[n+1, i] = u[n, i] + alpha * dt / dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        
        # Boundary conditions
        u[n+1, 0] = boundary_temp(n * dt)  # Left boundary
        u[n+1, Nx-1] = u[n+1, Nx-2]
    
    return u
