#include "CurveTendonForceLength.h"

CurveTendonForceLength::CurveTendonForceLength(
    double strain_at_one_norm_force,
    double stiffness_at_one_norm_force,
    double norm_force_at_toe_end,
    double curviness)
{
    if (strain_at_one_norm_force == 0.0) {
        // Default parameters
        strain_at_one_norm_force_ = 0.049;  // 4.9% strain matches in-vivo measurements
        
        // From TendonForceLengthCurve::ensureCurveUpToDate()
        double e0 = strain_at_one_norm_force_;
        
        // Assign the stiffness. By eyeball, this agrees well with various in-vitro tendon data.
        stiffness_at_one_norm_force_ = 1.375 / e0;
        norm_force_at_toe_end_ = 2.0 / 3.0;
        curviness_ = 0.5;
    } else {
        // Use provided values
        strain_at_one_norm_force_ = strain_at_one_norm_force;
        stiffness_at_one_norm_force_ = stiffness_at_one_norm_force;
        norm_force_at_toe_end_ = norm_force_at_toe_end;
        curviness_ = curviness;
    }
    
    // Build the curve
    curve_ = SmoothSegmentedFunction::createTendonForceLengthCurve(
        strain_at_one_norm_force_,
        stiffness_at_one_norm_force_,
        norm_force_at_toe_end_,
        curviness_,
        false
    );
}

double CurveTendonForceLength::calcValue(double x) const
{
    return curve_.calcValue(x);
}

double CurveTendonForceLength::calcDerivative(double x, int order) const
{
    return curve_.calcDerivative(x, order);
}

std::vector<double> CurveTendonForceLength::calcValDeriv(double x) const
{
    return curve_.calcValDeriv(x);
}