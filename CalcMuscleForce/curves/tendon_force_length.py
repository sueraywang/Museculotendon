from CalcMuscleForce.curves.curve_base import *
from CalcMuscleForce.utils.smooth_segmented import SmoothSegmentedFunction

class CurveTendonForceLength(CurveBase):
    """Curve for tendon force-length relationship"""
    
    def __init__(
        self,
        strain_at_one_norm_force: float = 0.0,
        stiffness_at_one_norm_force: float = 0.0,
        norm_force_at_toe_end: float = 0.0,
        curviness: float = 0.0
    ):
        """
        Initialize tendon force-length curve
        
        Args:
            strain_at_one_norm_force: Strain at one normalized force
            stiffness_at_one_norm_force: Stiffness at one normalized force
            norm_force_at_toe_end: Normalized force at toe end
            curviness: Curviness parameter (0-1)
        """
        if strain_at_one_norm_force == 0.0:
            # Default parameters
            self.strain_at_one_norm_force = 0.049  # 4.9% strain matches in-vivo measurements
            
            # From TendonForceLengthCurve::ensureCurveUpToDate()
            e0 = self.strain_at_one_norm_force
            
            # Assign the stiffness. By eyeball, this agrees well with various in-vitro tendon data.
            self.stiffness_at_one_norm_force = 1.375 / e0
            self.norm_force_at_toe_end = 2.0 / 3.0
            self.curviness = 0.5
        else:
            # Use provided values
            self.strain_at_one_norm_force = strain_at_one_norm_force
            self.stiffness_at_one_norm_force = stiffness_at_one_norm_force
            self.norm_force_at_toe_end = norm_force_at_toe_end
            self.curviness = curviness
        
        # Build the curve
        self.curve = SmoothSegmentedFunction.create_tendon_force_length_curve(
            self.strain_at_one_norm_force,
            self.stiffness_at_one_norm_force,
            self.norm_force_at_toe_end,
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