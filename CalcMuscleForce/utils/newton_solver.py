from typing import Tuple, Callable

class NewtonSolver:
    """Newton-Raphson solver for f(x) = 0"""
    
    @staticmethod
    def solve(
        func: Callable[[float], Tuple[float, float]],
        x0: float,
        tol: float = 1e-9,
        kmax: int = 100,
        dxmax: float = float('inf')
    ) -> float:
        """
        Solve f(x) = 0 using Newton's method
        
        Args:
            func: Function that returns (f(x), f'(x))
            x0: Initial guess
            tol: Tolerance for convergence
            kmax: Maximum number of iterations
            dxmax: Maximum step size
            
        Returns:
            Solution x where f(x) ≈ 0
        """
        x = x0
        k = 1
        
        while k < kmax:
            # Evaluate function and its Jacobian
            f, J = func(x)
            
            # Check for convergence in function value
            if abs(f) < tol:
                break
            
            # Calculate step
            dx = -f / J
            
            # Check for step size limit
            if abs(dx) > dxmax:
                dx = dxmax if dx > 0 else -dxmax
            
            # Update x
            x = x + dx
            
            # Check for convergence in step size
            if abs(dx) < tol:
                break
            
            k += 1
        
        return x