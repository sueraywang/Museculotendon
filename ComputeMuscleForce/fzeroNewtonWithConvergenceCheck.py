import numpy as np
from scipy.linalg import solve

def fzero_newton(fun, x0, tol=1e-9, kmax=100, dxmax=np.inf, verbose=False):
    """
    Newton-Raphson search for f(x) = 0 with convergence diagnostics.
    
    Parameters:
        fun (function): Function that returns a tuple (f, J) where f is the objective and J is its Jacobian.
        x0 (numpy.ndarray): Initial guess.
        tol (float, optional): Tolerance for convergence. Default is 1e-9.
        kmax (int, optional): Maximum number of iterations. Default is 100.
        dxmax (float, optional): Maximum step size allowed. Default is np.inf.
        verbose (bool, optional): If True, prints convergence information at each iteration. Default is False.
    
    Returns:
        x (numpy.ndarray): Solution.
        k (int): Number of iterations.
        dx (numpy.ndarray): Residual.
        convergence (bool): Whether the method converged.
    """
    k = 1
    x = np.array([x0])
    dx = np.zeros_like(x)
    convergence = False
    
    while k < kmax:
        f, J = fun(x)
        
        # Check if the function value is within the tolerance (convergence condition)
        norm_f = np.linalg.norm(f)
        if norm_f < tol:
            convergence = True
            break
    
        dx = -solve(J, f)
        
        # Limit the step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx > dxmax:
            if verbose:
                print(f"Iteration {k}: Step size too large, stopping. |dx| = {norm_dx:.3e}")
            break
        
        # Update x
        x = x + dx
        
        # Check if the step size is within the tolerance (convergence condition)
        if norm_dx < tol:
            convergence = True
            break
        
        # Optional: Print progress for monitoring convergence
        if verbose:
            print(f"Iteration {k}: |f(x)| = {norm_f:.3e}, |dx| = {norm_dx:.3e}")
        
        k += 1
    
    if verbose:
        if convergence:
            print(f"Converged in {k} iterations. |f(x)| = {norm_f:.3e}, |dx| = {norm_dx:.3e}")
        else:
            print(f"Did not converge after {k} iterations. Last |f(x)| = {norm_f:.3e}, |dx| = {norm_dx:.3e}")
    
    return x, k, dx, convergence
