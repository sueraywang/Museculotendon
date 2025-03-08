#ifndef QUINTIC_BEZIER_CURVE_H
#define QUINTIC_BEZIER_CURVE_H

#include <vector>
#include <array>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include "NewtonSolver.h"

class QuinticBezierCurve {
public:
    // Calculate control points for a corner (elbow) in the curve
    static std::vector<std::array<double, 2>> calcCornerControlPoints(
        double x0, double y0, double dydx0,
        double x1, double y1, double dydx1,
        double curviness);
    
    // Calculate the value of the curve at parameter u
    static double calcVal(double u, const std::vector<std::array<double, 2>>& pts);
    
    // Calculate derivatives with respect to u parameter
    static double calcDerivU(double u, const std::vector<std::array<double, 2>>& pts, int order);
    
    // Calculate derivatives dy/dx
    static double calcDerivDYDX(double u, const std::vector<double>& xpts, 
                               const std::vector<double>& ypts, int order);
    
    // Clamp u to [0,1]
    static double clampU(double u);
    
    // Compute u parameter for a given x value
    static double calcU(double ax, const std::vector<double>& bezier_pts_x);
    
    // Find the index of the Bezier section containing x
    static int calcIndex(double x, const std::vector<std::vector<double>>& bezierPtsX);
};

#endif // QUINTIC_BEZIER_CURVE_H