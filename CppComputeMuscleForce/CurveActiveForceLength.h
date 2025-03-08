#ifndef CURVE_ACTIVE_FORCE_LENGTH_H
#define CURVE_ACTIVE_FORCE_LENGTH_H

#include "CurveBase.h"
#include "SmoothSegmentedFunction.h"

class CurveActiveForceLength : public CurveBase {
public:
    CurveActiveForceLength(double min_active_norm_fiber_length = 0.0,
                          double transition_norm_fiber_length = 0.0,
                          double max_active_norm_fiber_length = 0.0,
                          double shallow_ascending_slope = 0.0,
                          double minimum_value = 0.0);
    
    double calcValue(double x) const override;
    double calcDerivative(double x, int order) const override;
    std::vector<double> calcValDeriv(double x) const override;
    
    // Getter for minimum active fiber length (for determining bounds)
    double getMinNormActiveFiberLength() const { return min_norm_active_fiber_length_; }
    
private:
    double min_norm_active_fiber_length_;
    double transition_norm_fiber_length_;
    double max_norm_active_fiber_length_;
    double shallow_ascending_slope_;
    double minimum_value_;
    
    SmoothSegmentedFunction curve_;
};

#endif // CURVE_ACTIVE_FORCE_LENGTH_H