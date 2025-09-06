from CalcMuscleForce.curves.curve_base import *
from CalcMuscleForce.utils.smooth_segmented import SmoothSegmentedFunction

class CurveActiveForceLength(CurveBase):
    """Curve for active force-length relationship"""
    
    def __init__(
        self,
        min_active_norm_fiber_length: float = 0.0,
        transition_norm_fiber_length: float = 0.0,
        max_active_norm_fiber_length: float = 0.0,
        shallow_ascending_slope: float = 0.0,
        minimum_value: float = 0.0
    ):
        """
        Initialize active force-length curve
        
        Args:
            min_active_norm_fiber_length: Minimum active normalized fiber length
            transition_norm_fiber_length: Transition normalized fiber length
            max_active_norm_fiber_length: Maximum active normalized fiber length
            shallow_ascending_slope: Slope of the shallow ascending region
            minimum_value: Minimum value
        """
        if min_active_norm_fiber_length == 0.0:
            # Use default values
            self.min_norm_active_fiber_length = 0.47 - 0.0259
            self.transition_norm_fiber_length = 0.73
            self.max_norm_active_fiber_length = 1.8123
            self.shallow_ascending_slope = 0.8616
            self.minimum_value = 0.0
        else:
            # Use provided values
            self.min_norm_active_fiber_length = min_active_norm_fiber_length
            self.transition_norm_fiber_length = transition_norm_fiber_length
            self.max_norm_active_fiber_length = max_active_norm_fiber_length
            self.shallow_ascending_slope = shallow_ascending_slope
            self.minimum_value = minimum_value
        
        # Build the curve
        self.curve = SmoothSegmentedFunction.create_fiber_active_force_length_curve(
            self.min_norm_active_fiber_length,
            self.transition_norm_fiber_length,
            1.0,
            self.max_norm_active_fiber_length,
            self.minimum_value,
            self.shallow_ascending_slope,
            1.0,
            False
        )
    
    def calc_value(self, x: float) -> float:
        """Calculate the value of the curve at a given point"""
        return self.curve.calc_value(x)
    
    def calc_derivative(self, x: float, order: int) -> float:
        """Calculate the derivative of the curve at a given point"""
        return self.curve.calc_derivative(x, order)
    
    def calc_val_deriv(self, x: float) -> List[float]:
        """Calculate both value and derivatives in one call (for performance)"""
        return self.curve.calc_val_deriv(x)
    
    def get_min_norm_active_fiber_length(self) -> float:
        """Getter for minimum active fiber length (for determining bounds)"""
        return self.min_norm_active_fiber_length