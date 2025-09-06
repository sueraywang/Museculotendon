from CalcMuscleForce.curves.curve_base import *
from CalcMuscleForce.utils.smooth_segmented import SmoothSegmentedFunction

class CurveForceVelocity(CurveBase):
    """Curve for force-velocity relationship"""
    
    def __init__(
        self,
        concentric_slope_at_vmax: float = 0.0,
        concentric_slope_near_vmax: float = 0.0,
        isometric_slope: float = 0.0,
        eccentric_slope_at_vmax: float = 0.0,
        eccentric_slope_near_vmax: float = 0.0,
        max_eccentric_velocity_force_multiplier: float = 0.0,
        concentric_curviness: float = 0.0,
        eccentric_curviness: float = 0.0
    ):
        """
        Initialize force-velocity curve
        
        Args:
            concentric_slope_at_vmax: Slope at maximum concentric velocity
            concentric_slope_near_vmax: Slope near maximum concentric velocity
            isometric_slope: Slope at isometric velocity
            eccentric_slope_at_vmax: Slope at maximum eccentric velocity
            eccentric_slope_near_vmax: Slope near maximum eccentric velocity
            max_eccentric_velocity_force_multiplier: Force multiplier at maximum eccentric velocity
            concentric_curviness: Curviness parameter for concentric portion (0-1)
            eccentric_curviness: Curviness parameter for eccentric portion (0-1)
        """
        if concentric_slope_at_vmax == 0.0:
            # Default parameters
            self.concentric_slope_at_vmax = 0.0
            self.concentric_slope_near_vmax = 0.25
            self.isometric_slope = 5.0
            self.eccentric_slope_at_vmax = 0.0
            self.eccentric_slope_near_vmax = 0.15  # guess for now
            self.max_eccentric_velocity_force_multiplier = 1.4
            self.concentric_curviness = 0.6
            self.eccentric_curviness = 0.9
        else:
            # Use provided values
            self.concentric_slope_at_vmax = concentric_slope_at_vmax
            self.concentric_slope_near_vmax = concentric_slope_near_vmax
            self.isometric_slope = isometric_slope
            self.eccentric_slope_at_vmax = eccentric_slope_at_vmax
            self.eccentric_slope_near_vmax = eccentric_slope_near_vmax
            self.max_eccentric_velocity_force_multiplier = max_eccentric_velocity_force_multiplier
            self.concentric_curviness = concentric_curviness
            self.eccentric_curviness = eccentric_curviness
        
        # Build the curve
        self.curve = SmoothSegmentedFunction.create_fiber_force_velocity_curve(
            self.max_eccentric_velocity_force_multiplier,
            self.concentric_slope_at_vmax,
            self.concentric_slope_near_vmax,
            self.isometric_slope,
            self.eccentric_slope_at_vmax,
            self.eccentric_slope_near_vmax,
            self.concentric_curviness,
            self.eccentric_curviness,
            False
        )
    
    def calc_value(self, x: float) -> float:
        """Calculate the value of the curve at a given point"""
        return self.curve.calc_value(x)
    
    def calc_derivative(self, x: float, order: int) -> float:
        """Calculate the derivative of the curve at a given point"""
        if order < 0 or order > 2:
            raise ValueError("Order must be 0, 1, or 2")
        return self.curve.calc_derivative(x, order)
    
    def calc_val_deriv(self, x: float) -> List[float]:
        """Calculate both value and derivatives in one call (for performance)"""
        return self.curve.calc_val_deriv(x)