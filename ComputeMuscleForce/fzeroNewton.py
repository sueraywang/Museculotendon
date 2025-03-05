import numpy as np
from scipy.linalg import solve

def fzero_newton(fun, x0, tol=1e-9, kmax=100, dxmax=np.inf):
    """
    Newton-Raphson search for f(x) = 0.
    
    Parameters:
        fun (function): Function that returns a tuple (f, J) where f is the objective and J is its Jacobian.
        x0 (numpy.ndarray): Initial guess.
        tol (float, optional): Tolerance for convergence. Default is 1e-9.
        kmax (int, optional): Maximum number of iterations. Default is 100.
        dxmax (float, optional): Maximum step size allowed. Default is np.inf.
    
    Returns:
        x (numpy.ndarray): Solution.
        k (int): Number of iterations.
        dx (numpy.ndarray): Residual.
    """
    k = 1
    x = np.array(x0)
    dx = np.zeros_like(x)
    
    while k < kmax:
        f, J = fun(x)
        if np.linalg.norm(f) < tol:
            break
        dx = -solve(J, f)
        if np.linalg.norm(dx) > dxmax:
            break
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
        k += 1
    
    return x, k, dx
