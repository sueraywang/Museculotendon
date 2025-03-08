#include "CurveActiveForceLength.h"

CurveActiveForceLength::CurveActiveForceLength(
    double min_active_norm_fiber_length,
    double transition_norm_fiber_length,
    double max_active_norm_fiber_length,
    double shallow_ascending_slope,
    double minimum_value)
{
    if (min_active_norm_fiber_length == 0.0) {
        // Use default values
        min_norm_active_fiber_length_ = 0.47 - 0.0259;
        transition_norm_fiber_length_ = 0.73;
        max_norm_active_fiber_length_ = 1.8123;
        shallow_ascending_slope_ = 0.8616;
        minimum_value_ = 0.0;
    } else {
        // Use provided values
        min_norm_active_fiber_length_ = min_active_norm_fiber_length;
        transition_norm_fiber_length_ = transition_norm_fiber_length;
        max_norm_active_fiber_length_ = max_active_norm_fiber_length;
        shallow_ascending_slope_ = shallow_ascending_slope;
        minimum_value_ = minimum_value;
    }
    
    // Build the curve
    curve_ = SmoothSegmentedFunction::createFiberActiveForceLengthCurve(
        min_norm_active_fiber_length_,
        transition_norm_fiber_length_,
        1.0,
        max_norm_active_fiber_length_,
        minimum_value_,
        shallow_ascending_slope_,
        1.0,
        false
    );
}

double CurveActiveForceLength::calcValue(double x) const
{
    return curve_.calcValue(x);
}

double CurveActiveForceLength::calcDerivative(double x, int order) const
{
    return curve_.calcDerivative(x, order);
}

std::vector<double> CurveActiveForceLength::calcValDeriv(double x) const
{
    return curve_.calcValDeriv(x);
}