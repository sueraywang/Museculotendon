#ifndef CURVE_FIBER_FORCE_LENGTH_H
#define CURVE_FIBER_FORCE_LENGTH_H

#include "CurveBase.h"
#include "SmoothSegmentedFunction.h"

class CurveFiberForceLength : public CurveBase {
public:
    CurveFiberForceLength(double strain_at_zero_force = 0.0,
                         double strain_at_one_norm_force = 0.0,
                         double stiffness_at_low_force = 0.0,
                         double stiffness_at_one_norm_force = 0.0,
                         double curviness = 0.0);
    
    double calcValue(double x) const override;
    double calcDerivative(double x, int order) const override;
    std::vector<double> calcValDeriv(double x) const override;
    
private:
    double strain_at_zero_force_;
    double strain_at_one_norm_force_;
    double stiffness_at_low_force_;
    double stiffness_at_one_norm_force_;
    double curviness_;
    
    SmoothSegmentedFunction curve_;
};

#endif // CURVE_FIBER_FORCE_LENGTH_H