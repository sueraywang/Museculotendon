#include "CurveForceVelocity.h"

CurveForceVelocity::CurveForceVelocity(
    double concentric_slope_at_vmax,
    double concentric_slope_near_vmax,
    double isometric_slope,
    double eccentric_slope_at_vmax,
    double eccentric_slope_near_vmax,
    double max_eccentric_velocity_force_multiplier,
    double concentric_curviness,
    double eccentric_curviness)
{
    if (concentric_slope_at_vmax == 0.0) {
        // Default parameters
        concentric_slope_at_vmax_ = 0.0;
        concentric_slope_near_vmax_ = 0.25;
        isometric_slope_ = 5.0;
        eccentric_slope_at_vmax_ = 0.0;
        eccentric_slope_near_vmax_ = 0.15;  // guess for now
        max_eccentric_velocity_force_multiplier_ = 1.4;
        concentric_curviness_ = 0.6;
        eccentric_curviness_ = 0.9;
    } else {
        // Use provided values
        concentric_slope_at_vmax_ = concentric_slope_at_vmax;
        concentric_slope_near_vmax_ = concentric_slope_near_vmax;
        isometric_slope_ = isometric_slope;
        eccentric_slope_at_vmax_ = eccentric_slope_at_vmax;
        eccentric_slope_near_vmax_ = eccentric_slope_near_vmax;
        max_eccentric_velocity_force_multiplier_ = max_eccentric_velocity_force_multiplier;
        concentric_curviness_ = concentric_curviness;
        eccentric_curviness_ = eccentric_curviness;
    }
    
    // Build the curve
    curve_ = SmoothSegmentedFunction::createFiberForceVelocityCurve(
        max_eccentric_velocity_force_multiplier_,
        concentric_slope_at_vmax_,
        concentric_slope_near_vmax_,
        isometric_slope_,
        eccentric_slope_at_vmax_,
        eccentric_slope_near_vmax_,
        concentric_curviness_,
        eccentric_curviness_,
        false
    );
}

double CurveForceVelocity::calcValue(double x) const
{
    return curve_.calcValue(x);
}

double CurveForceVelocity::calcDerivative(double x, int order) const
{
    if (order < 0 || order > 2) {
        throw std::invalid_argument("Order must be 0, 1, or 2");
    }
    return curve_.calcDerivative(x, order);
}

std::vector<double> CurveForceVelocity::calcValDeriv(double x) const
{
    return curve_.calcValDeriv(x);
}