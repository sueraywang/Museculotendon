#ifndef CURVE_BASE_H
#define CURVE_BASE_H

#include <vector>

// Base class for all muscle curves
class CurveBase {
public:
    virtual ~CurveBase() = default;
    
    // Calculate the value of the curve at a given point
    virtual double calcValue(double x) const = 0;
    
    // Calculate the derivative of the curve at a given point
    virtual double calcDerivative(double x, int order) const = 0;
    
    // Calculate both value and derivatives in one call (for performance)
    virtual std::vector<double> calcValDeriv(double x) const = 0;
};

#endif // CURVE_BASE_H