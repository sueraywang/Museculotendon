#ifndef CURVE_TENDON_FORCE_LENGTH_H
#define CURVE_TENDON_FORCE_LENGTH_H

#include "CurveBase.h"
#include "SmoothSegmentedFunction.h"

class CurveTendonForceLength : public CurveBase {
public:
    CurveTendonForceLength(double strain_at_one_norm_force = 0.0,
                          double stiffness_at_one_norm_force = 0.0,
                          double norm_force_at_toe_end = 0.0,
                          double curviness = 0.0);
    
    double calcValue(double x) const override;
    double calcDerivative(double x, int order) const override;
    std::vector<double> calcValDeriv(double x) const override;
    
private:
    double strain_at_one_norm_force_;
    double stiffness_at_one_norm_force_;
    double norm_force_at_toe_end_;
    double curviness_;
    
    SmoothSegmentedFunction curve_;
};

#endif // CURVE_TENDON_FORCE_LENGTH_H