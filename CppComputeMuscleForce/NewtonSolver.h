#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include <functional>
#include <limits>
#include <cmath>
#include <utility>

class NewtonSolver {
public:
    // Newton-Raphson solver for f(x) = 0
    // Objective function should return a pair of (f(x), f'(x))
    template<typename ObjectiveFunc>
    double solve(ObjectiveFunc func, double x0, 
                 double tol = 1e-9, int kmax = 100, 
                 double dxmax = std::numeric_limits<double>::infinity()) const {
        double x = x0;
        double dx = 0.0;
        int k = 1;
        
        while (k < kmax) {
            // Evaluate function and its Jacobian
            auto result = func(x);
            double f = result.first;
            double J = result.second;
            
            // Check for convergence in function value
            if (std::abs(f) < tol) {
                break;
            }
            
            // Calculate step
            dx = -f / J;
            
            // Check for step size limit
            if (std::abs(dx) > dxmax) {
                dx = (dx > 0 ? dxmax : -dxmax);
            }
            
            // Update x
            x = x + dx;
            
            // Check for convergence in step size
            if (std::abs(dx) < tol) {
                break;
            }
            
            ++k;
        }
        
        return x;
    }
};

#endif // NEWTON_SOLVER_H