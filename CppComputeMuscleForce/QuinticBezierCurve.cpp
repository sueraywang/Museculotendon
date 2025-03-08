#include "QuinticBezierCurve.h"

std::vector<std::array<double, 2>> QuinticBezierCurve::calcCornerControlPoints(
    double x0, double y0, double dydx0,
    double x1, double y1, double dydx1,
    double curviness)
{
    double root_eps = std::sqrt(std::numeric_limits<double>::epsilon());
    double xC;
    
    if (std::abs(dydx0 - dydx1) > root_eps) {
        xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1);
    } else {
        xC = (x1 + x0) / 2;
    }
    
    double yC = (xC - x1) * dydx1 + y1;
    
    std::vector<std::array<double, 2>> xy_pts(6);
    
    xy_pts[0] = {x0, y0};
    xy_pts[5] = {x1, y1};
    
    xy_pts[1] = {x0 + curviness * (xC - xy_pts[0][0]), y0 + curviness * (yC - xy_pts[0][1])};
    xy_pts[2] = xy_pts[1];
    
    xy_pts[3] = {xy_pts[5][0] + curviness * (xC - xy_pts[5][0]), xy_pts[5][1] + curviness * (yC - xy_pts[5][1])};
    xy_pts[4] = xy_pts[3];
    
    return xy_pts;
}

double QuinticBezierCurve::calcVal(double u, const std::vector<std::array<double, 2>>& pts)
{
    double u5 = 1;
    double u4 = u;
    double u3 = u4 * u;
    double u2 = u3 * u;
    double u1 = u2 * u;
    double u0 = u1 * u;
    
    double t2 = u1 * 5;
    double t3 = u2 * 10;
    double t4 = u3 * 10;
    double t5 = u4 * 5;
    double t9 = u0 * 5;
    double t10 = u1 * 20;
    double t11 = u2 * 30;
    double t15 = u0 * 10;
    
    double val =
        pts[0][1] * (u0 * -1 + t2 - t3 + t4 - t5 + u5 * 1) +
        pts[1][1] * (t9 - t10 + t11 + u3 * -20 + t5) +
        pts[2][1] * (-t15 + u1 * 30 - t11 + t4) +
        pts[3][1] * (t15 - t10 + t3) +
        pts[4][1] * (-t9 + t2) +
        pts[5][1] * u0 * 1;
    
    return val;
}

double QuinticBezierCurve::calcDerivU(double u, const std::vector<std::array<double, 2>>& pts, int order)
{
    double p0 = pts[0][1];
    double p1 = pts[1][1];
    double p2 = pts[2][1];
    double p3 = pts[3][1];
    double p4 = pts[4][1];
    double p5 = pts[5][1];
    double val = 0.0;
    
    switch (order) {
        case 1: {
            double t1 = u * u;  // u ^ 2
            double t2 = t1 * t1;  // t1 ^ 2
            double t4 = t1 * u;
            double t5 = t4 * 20.0;
            double t6 = t1 * 30.0;
            double t7 = u * 20.0;
            double t10 = t2 * 25.0;
            double t11 = t4 * 80.0;
            double t12 = t1 * 90.0;
            double t16 = t2 * 50.0;
            val = p0 * (t2 * (-5.0) + t5 - t6 + t7 - 5.0) +
                  p1 * (t10 - t11 + t12 + u * (-40.0) + 5.0) +
                  p2 * (-t16 + t4 * 120.0 - t12 + t7) +
                  p3 * (t16 - t11 + t6) +
                  p4 * (-t10 + t5) +
                  p5 * t2 * 5.0;
            break;
        }
        case 2: {
            double t1 = u * u;  // u ^ 2
            double t2 = t1 * u;
            double t4 = t1 * 60.0;
            double t5 = u * 60.0;
            double t8 = t2 * 100.0;
            double t9 = t1 * 240.0;
            double t10 = u * 180.0;
            double t13 = t2 * 200.0;
            val = p0 * (t2 * (-20.0) + t4 - t5 + 20.0) +
                  p1 * (t8 - t9 + t10 - 40.0) +
                  p2 * (-t13 + t1 * 360.0 - t10 + 20.0) +
                  p3 * (t13 - t9 + t5) +
                  p4 * (-t8 + t4) +
                  p5 * t2 * 20.0;
            break;
        }
        case 3: {
            double t1 = u * u;  // u ^ 2
            double t3 = u * 120.0;
            double t6 = t1 * 300.0;
            double t7 = u * 480.0;
            double t10 = t1 * 600.0;
            val = p0 * (t1 * (-60.0) + t3 - 60.0) +
                  p1 * (t6 - t7 + 180.0) +
                  p2 * (-t10 + u * 720.0 - 180.0) +
                  p3 * (t10 - t7 + 60.0) +
                  p4 * (-t6 + t3) +
                  p5 * t1 * 60.0;
            break;
        }
        case 4: {
            double t4 = u * 600.0;
            double t7 = u * 1200.0;
            val = p0 * (u * (-120.0) + 120.0) +
                  p1 * (t4 - 480.0) +
                  p2 * (-t7 + 720.0) +
                  p3 * (t7 - 480.0) +
                  p4 * (-t4 + 120.0) +
                  p5 * u * 120.0;
            break;
        }
        case 5: {
            val = p0 * (-120.0) +
                  p1 * 600.0 +
                  p2 * (-1200.0) +
                  p3 * 1200.0 +
                  p4 * (-600.0) +
                  p5 * 120.0;
            break;
        }
        default:
            val = 0;
    }
    
    return val;
}

double QuinticBezierCurve::calcDerivDYDX(double u, const std::vector<double>& xpts, 
                                       const std::vector<double>& ypts, int order)
{
    double val = 0.0;
    
    // Create points array for x and y separately
    std::vector<std::array<double, 2>> xptsArray(6);
    std::vector<std::array<double, 2>> yptsArray(6);
    
    for (int i = 0; i < 6; ++i) {
        xptsArray[i] = {0.0, xpts[i]};
        yptsArray[i] = {0.0, ypts[i]};
    }
    
    // Compute the derivative d^n y / dx^n
    if (order == 1) {  // Calculate dy/dx
        double dxdu = calcDerivU(u, xptsArray, 1);
        double dydu = calcDerivU(u, yptsArray, 1);
        val = dydu / dxdu;
    } else if (order == 2) {  // Calculate d^2y/dx^2
        double dxdu = calcDerivU(u, xptsArray, 1);
        double dydu = calcDerivU(u, yptsArray, 1);
        double d2xdu2 = calcDerivU(u, xptsArray, 2);
        double d2ydu2 = calcDerivU(u, yptsArray, 2);
        double t1 = 1.0 / dxdu;
        double t3 = dxdu * dxdu;  // dxdu ^ 2
        val = (d2ydu2 * t1 - dydu / t3 * d2xdu2) * t1;
    } else if (order == 3) {  // Calculate d^3y/dx^3
        double dxdu = calcDerivU(u, xptsArray, 1);
        double dydu = calcDerivU(u, yptsArray, 1);
        double d2xdu2 = calcDerivU(u, xptsArray, 2);
        double d2ydu2 = calcDerivU(u, yptsArray, 2);
        double d3xdu3 = calcDerivU(u, xptsArray, 3);
        double d3ydu3 = calcDerivU(u, yptsArray, 3);
        double t1 = 1.0 / dxdu;
        double t3 = dxdu * dxdu;  // dxdu ^ 2
        double t4 = 1.0 / t3;
        double t11 = d2xdu2 * d2xdu2;  // (d2xdu2 ^ 2)
        double t14 = dydu * t4;
        val = ((d3ydu3 * t1 - 2 * d2ydu2 * t4 * d2xdu2
              + 2 * dydu / t3 / dxdu * t11 - t14 * d3xdu3) * t1
              - (d2ydu2 * t1 - t14 * d2xdu2) * t4 * d2xdu2) * t1;
    } else if (order > 3) {
        // Higher order derivatives are complex and not needed for most applications
        // Implement as needed
        throw std::runtime_error("Derivatives of order > 3 not implemented yet");
    }
    
    return val;
}

double QuinticBezierCurve::clampU(double u)
{
    return std::max(0.0, std::min(1.0, u));
}

double QuinticBezierCurve::calcU(double ax, const std::vector<double>& bezier_pts_x)
{
    // Check to make sure that ax is in the curve domain
    double min_x = *std::min_element(bezier_pts_x.begin(), bezier_pts_x.end());
    double max_x = *std::max_element(bezier_pts_x.begin(), bezier_pts_x.end());
    
    if (!(min_x <= ax && ax <= max_x)) {
        throw std::runtime_error("Input ax is not in the domain of the Bezier curve specified by the control points");
    }
    
    // Create points array
    std::vector<std::array<double, 2>> pts(6);
    for (size_t i = 0; i < 6; ++i) {
        pts[i] = {0.0, bezier_pts_x[i]};
    }
    
    // Define the objective function
    auto objective = [&pts, ax](double u) -> std::pair<double, double> {
        double x = QuinticBezierCurve::calcVal(u, pts);
        double dxdu = QuinticBezierCurve::calcDerivU(u, pts, 1);
        return std::make_pair(x - ax, dxdu);
    };
    
    // Initial guess
    double u_init = (ax - min_x) / (max_x - min_x);
    
    // Use Newton's method to find the root
    NewtonSolver solver;
    return solver.solve(objective, u_init, 1e-9, 10);
}

int QuinticBezierCurve::calcIndex(double x, const std::vector<std::vector<double>>& bezierPtsX)
{
    int idx = 0;
    bool flag_found = false;
    
    // Iterate over columns in bezier_pts_x
    for (size_t i = 0; i < bezierPtsX.size(); ++i) {
        if (bezierPtsX[i][0] <= x && x < bezierPtsX[i][5]) {
            idx = i;
            flag_found = true;
            break;
        }
    }
    
    // Check if x is identically the last point
    if (!flag_found && x == bezierPtsX.back()[5]) {
        idx = bezierPtsX.size() - 1;
        flag_found = true;
    }
    
    // Optional: Raise an error if the value x is not within the Bezier curve set
    if (!flag_found) {
        throw std::runtime_error("A value of x was used that is not within the Bezier curve set");
    }
    
    return idx;
}