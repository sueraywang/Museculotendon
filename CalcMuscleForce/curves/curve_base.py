from typing import List

class CurveBase:
    """Base class for all muscle curves"""
    
    def calc_value(self, x: float) -> float:
        """Calculate the value of the curve at a given point"""
        raise NotImplementedError
    
    def calc_derivative(self, x: float, order: int) -> float:
        """Calculate the derivative of the curve at a given point"""
        raise NotImplementedError
    
    def calc_val_deriv(self, x: float) -> List[float]:
        """Calculate both value and derivatives in one call (for performance)"""
        raise NotImplementedError