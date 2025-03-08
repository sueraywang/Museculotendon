#include "CurveFiberForceLength.h"

CurveFiberForceLength::CurveFiberForceLength(
    double strain_at_zero_force,
    double strain_at_one_norm_force,
    double stiffness_at_low_force,
    double stiffness_at_one_norm_force,
    double curviness)
{
    if (strain_at_zero_force == 0.0 && strain_at_one_norm_force == 0.0) {
        // Default parameters
        strain_at_zero_force_ = 0.0;
        strain_at_one_norm_force_ = 0.7;
        
        double e0 = strain_at_zero_force_;  // properties of reference curve
        double e1 = strain_at_one_norm_force_;
        
        // Assign the stiffnesses. These values are based on the Thelen2003
        // default curve and the EDL passive force-length curve found
        // experimentally by Winters, Takahashi, Ward, and Lieber (2011).
        stiffness_at_one_norm_force_ = 2.0 / (e1 - e0);
        stiffness_at_low_force_ = 0.2;
        
        // Fit the curviness parameter to the reference curve
        curviness_ = 0.75;
    } else {
        // Use provided values
        strain_at_zero_force_ = strain_at_zero_force;
        strain_at_one_norm_force_ = strain_at_one_norm_force;
        stiffness_at_low_force_ = stiffness_at_low_force;
        stiffness_at_one_norm_force_ = stiffness_at_one_norm_force;
        curviness_ = curviness;
    }
    
    // Build the curve
    curve_ = SmoothSegmentedFunction::createFiberForceLengthCurve(
        strain_at_zero_force_,
        strain_at_one_norm_force_,
        stiffness_at_low_force_,
        stiffness_at_one_norm_force_,
        curviness_,
        false
    );
}

double CurveFiberForceLength::calcValue(double x) const
{
    return curve_.calcValue(x);
}

double CurveFiberForceLength::calcDerivative(double x, int order) const
{
    return curve_.calcDerivative(x, order);
}

std::vector<double> CurveFiberForceLength::calcValDeriv(double x) const
{
    return curve_.calcValDeriv(x);
}