#ifndef CURVE_FORCE_VELOCITY_H
#define CURVE_FORCE_VELOCITY_H

#include "CurveBase.h"
#include "SmoothSegmentedFunction.h"

class CurveForceVelocity : public CurveBase {
public:
    CurveForceVelocity(double concentric_slope_at_vmax = 0.0,
                      double concentric_slope_near_vmax = 0.0,
                      double isometric_slope = 0.0,
                      double eccentric_slope_at_vmax = 0.0,
                      double eccentric_slope_near_vmax = 0.0,
                      double max_eccentric_velocity_force_multiplier = 0.0,
                      double concentric_curviness = 0.0,
                      double eccentric_curviness = 0.0);
    
    double calcValue(double x) const override;
    double calcDerivative(double x, int order) const override;
    std::vector<double> calcValDeriv(double x) const override;
    
private:
    double concentric_slope_at_vmax_;
    double concentric_slope_near_vmax_;
    double isometric_slope_;
    double eccentric_slope_at_vmax_;
    double eccentric_slope_near_vmax_;
    double max_eccentric_velocity_force_multiplier_;
    double concentric_curviness_;
    double eccentric_curviness_;
    
    SmoothSegmentedFunction curve_;
};

#endif // CURVE_FORCE_VELOCITY_H