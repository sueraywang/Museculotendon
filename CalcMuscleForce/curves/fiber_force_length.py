from CalcMuscleForce.curves.curve_base import *
from CalcMuscleForce.utils.smooth_segmented import SmoothSegmentedFunction

class CurveFiberForceLength(CurveBase):
    """Curve for fiber force-length relationship (passive component)"""
    
    def __init__(
        self,
        strain_at_zero_force: float = 0.0,
        strain_at_one_norm_force: float = 0.0,
        stiffness_at_low_force: float = 0.0,
        stiffness_at_one_norm_force: float = 0.0,
        curviness: float = 0.0
    ):
        """
        Initialize fiber force-length curve
        
        Args:
            strain_at_zero_force: Strain at zero force
            strain_at_one_norm_force: Strain at one normalized force
            stiffness_at_low_force: Stiffness at low force
            stiffness_at_one_norm_force: Stiffness at one normalized force
            curviness: Curviness parameter (0-1)
        """
        if strain_at_zero_force == 0.0 and strain_at_one_norm_force == 0.0:
            # Default parameters
            self.strain_at_zero_force = 0.0
            self.strain_at_one_norm_force = 0.7
            
            e0 = self.strain_at_zero_force  # properties of reference curve
            e1 = self.strain_at_one_norm_force
            
            # Assign the stiffnesses. These values are based on the Thelen2003
            # default curve and the EDL passive force-length curve found
            # experimentally by Winters, Takahashi, Ward, and Lieber (2011).
            self.stiffness_at_one_norm_force = 2.0 / (e1 - e0)
            self.stiffness_at_low_force = 0.2
            
            # Fit the curviness parameter to the reference curve
            self.curviness = 0.75
        else:
            # Use provided values
            self.strain_at_zero_force = strain_at_zero_force
            self.strain_at_one_norm_force = strain_at_one_norm_force
            self.stiffness_at_low_force = stiffness_at_low_force
            self.stiffness_at_one_norm_force = stiffness_at_one_norm_force
            self.curviness = curviness
        
        # Build the curve
        self.curve = SmoothSegmentedFunction.create_fiber_force_length_curve(
            self.strain_at_zero_force,
            self.strain_at_one_norm_force,
            self.stiffness_at_low_force,
            self.stiffness_at_one_norm_force,
            self.curviness,
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