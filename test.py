from fzeroNewtonWithConvergenceCheck import *

# Example function and its Jacobian
def fun(x):
    f = np.array([x[0]**2 - 2])
    J = np.array([[2*x[0]]])
    return f, J

# Call Newton's method with verbose output
x0 = [1.0]
x, k, dx, converged = fzero_newton(fun, x0, verbose=True)
