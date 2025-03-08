#ifndef SMOOTH_SEGMENTED_FUNCTION_H
#define SMOOTH_SEGMENTED_FUNCTION_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include "QuinticBezierCurve.h"

class SmoothSegmentedFunction {
public:
    // Default constructor - creates an empty, invalid function
    // This is only used for member variable initialization and should be
    // replaced with a properly constructed function before use
    SmoothSegmentedFunction() 
        : mXVec_(0), mYVec_(0), x0_(0), x1_(0), y0_(0), y1_(0),
          dydx0_(0), dydx1_(0), computeIntegral_(false), 
          intx0x1_(false), numBezierSections_(0) {}
          
    SmoothSegmentedFunction(const std::vector<std::vector<double>>& mX,
                           const std::vector<std::vector<double>>& mY,
                           double x0, double x1, double y0, double y1,
                           double dydx0, double dydx1,
                           bool computeIntegral = false, bool intx0x1 = true);
    
    // Calculate the value at a given point
    double calcValue(double x) const;
    
    // Calculate derivative of specified order
    double calcDerivative(double x, int order) const;
    
    // Calculate value and derivatives (up to 2nd) at once
    std::vector<double> calcValDeriv(double x) const;
    
    // Get the domain boundaries
    std::vector<double> getDomains() const;
    
    // Static factory methods for creating various types of curves
    static SmoothSegmentedFunction createFiberForceLengthCurve(
        double eZero, double eIso, double kLow, double kIso, 
        double curviness, bool computeIntegral);
    
    static SmoothSegmentedFunction createTendonForceLengthCurve(
        double eIso, double kIso, double fToe, double curviness, 
        bool computeIntegral);
    
    static SmoothSegmentedFunction createFiberForceVelocityCurve(
        double fmaxE, double dydxC, double dydxNearC, double dydxIso,
        double dydxE, double dydxNearE, double concCurviness, 
        double eccCurviness, bool computeIntegral);
    
    static SmoothSegmentedFunction createFiberForceVelocityInverseCurve(
        double fmaxE, double dydxC, double dydxNearC, double dydxIso,
        double dydxE, double dydxNearE, double concCurviness, 
        double eccCurviness, bool computeIntegral);
    
    static SmoothSegmentedFunction createFiberActiveForceLengthCurve(
        double x0, double x1, double x2, double x3, double ylow,
        double dydx, double curviness, bool computeIntegral);
    
private:
    // Helper function to scale curviness
    static double scaleCurviness(double curviness);
    
    // Data members
    std::vector<std::vector<double>> mXVec_;
    std::vector<std::vector<double>> mYVec_;
    double x0_;
    double x1_;
    double y0_;
    double y1_;
    double dydx0_;
    double dydx1_;
    bool computeIntegral_;
    bool intx0x1_;
    int numBezierSections_;
};

#endif // SMOOTH_SEGMENTED_FUNCTION_H