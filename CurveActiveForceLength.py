import numpy as np
from CurveBase import CurveBase
from SmoothSegmentedFunction import SmoothSegmentedFunction

class CurveActiveForceLength(CurveBase):
    def __init__(self, min_active_norm_fiber_length=None, transition_norm_fiber_length=None,
                 max_active_norm_fiber_length=None, shallow_ascending_slope=None, minimum_value=None):
        if min_active_norm_fiber_length is None:
            min_active_norm_fiber_length = 0.47 - 0.0259
            transition_norm_fiber_length = 0.73
            max_active_norm_fiber_length = 1.8123
            shallow_ascending_slope = 0.8616
            minimum_value = 0.0
        elif not all(v is not None for v in [min_active_norm_fiber_length, transition_norm_fiber_length,
                                             max_active_norm_fiber_length, shallow_ascending_slope, minimum_value]):
            raise ValueError('Please provide all five arguments if not using default values.')

        self.min_norm_active_fiber_length = min_active_norm_fiber_length
        self.transition_norm_fiber_length = transition_norm_fiber_length
        self.max_norm_active_fiber_length = max_active_norm_fiber_length
        self.shallow_ascending_slope = shallow_ascending_slope
        self.minimum_value = minimum_value

        # Build the curve
        self.m_curve = SmoothSegmentedFunction.createFiberActiveForceLengthCurve(
            self.min_norm_active_fiber_length,
            self.transition_norm_fiber_length,
            1.0,
            self.max_norm_active_fiber_length,
            self.minimum_value,
            self.shallow_ascending_slope,
            1.0,
            compute_integral=False
        )

    @staticmethod
    def test():
        x = np.linspace(0.0, 2.0, 1000)
        curve = CurveActiveForceLength()
        cname = 'Active Force Length Curve'
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y0, y1, y2 = CurveBase.test(curve, cname, x)
        # Opening a file
        file = open('ActiveForceLengthData.npy', 'w')
        # Save the array to a binary file
        np.save('ActiveForceLengthData.npy', [y0,y1,y2])
        file.close()

CurveActiveForceLength.test()