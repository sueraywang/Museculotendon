import numpy as np
from CurveBase import CurveBase
from SmoothSegmentedFunction import SmoothSegmentedFunction

class CurveForceVelocityInverse(CurveBase):
    """
    CurveForceVelocityInverse
    Doxygen API: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1ForceVelocityInverseCurve.html
    Source: https://github.com/opensim-org/opensim-core/blob/master/OpenSim/Actuators/ForceVelocityInverseCurve.h
    """

    def __init__(self, concentric_slope_at_vmax=None, concentric_slope_near_vmax=None,
                 isometric_slope=None, eccentric_slope_at_vmax=None,
                 eccentric_slope_near_vmax=None, max_eccentric_velocity_force_multiplier=None,
                 concentric_curviness=None, eccentric_curviness=None):
        """
        This class serves as a serializable ForceVelocityInverseCurve for use in
        equilibrium muscle models. The inverse force-velocity curve is
        dimensionless: force is normalized to maximum isometric force and velocity
        is normalized to the maximum muscle contraction velocity (vmax), where vmax
        is expressed in units of optimal_fiber_lengths per second. Negative
        normalized velocities correspond to concentric contraction (i.e.,
        shortening). The inverse force-velocity curve is constructed from 8
        properties, which are identical to those used to construct the
        corresponding force-velocity curve. See ForceVelocityCurve for descriptions
        of these parameters.
        """

        if concentric_slope_at_vmax is None and concentric_slope_near_vmax is None and \
           isometric_slope is None and eccentric_slope_at_vmax is None and \
           eccentric_slope_near_vmax is None and max_eccentric_velocity_force_multiplier is None and \
           concentric_curviness is None and eccentric_curviness is None:
            # Default parameters
            concentric_slope_at_vmax = 0.1
            concentric_slope_near_vmax = 0.25
            isometric_slope = 5.0
            eccentric_slope_at_vmax = 0.1
            eccentric_slope_near_vmax = 0.15
            max_eccentric_velocity_force_multiplier = 1.4
            concentric_curviness = 0.6
            eccentric_curviness = 0.9
        elif all(v is not None for v in [concentric_slope_at_vmax, concentric_slope_near_vmax,
                                         isometric_slope, eccentric_slope_at_vmax,
                                         eccentric_slope_near_vmax, max_eccentric_velocity_force_multiplier,
                                         concentric_curviness, eccentric_curviness]):
            pass
        else:
            raise ValueError("Please provide either 0 or 8 arguments")

        self.concentric_slope_at_vmax = concentric_slope_at_vmax
        self.concentric_slope_near_vmax = concentric_slope_near_vmax
        self.isometric_slope = isometric_slope
        self.eccentric_slope_at_vmax = eccentric_slope_at_vmax
        self.eccentric_slope_near_vmax = eccentric_slope_near_vmax
        self.max_eccentric_velocity_force_multiplier = max_eccentric_velocity_force_multiplier
        self.concentric_curviness = concentric_curviness
        self.eccentric_curviness = eccentric_curviness

        # From ForceVelocityInverseCurve::buildCurve
        self.m_curve = SmoothSegmentedFunction.createFiberForceVelocityInverseCurve(
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

    @staticmethod
    def test():
        curve = CurveForceVelocityInverse()
        cname = 'Force Velocity Curve'
        x = np.linspace(0.05, 1.35, 1000)
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y0, y1, y2 = CurveBase.test(curve, cname, x)
        # Opening a file
        file = open('ForceVelocityInverseData.npy', 'w')
        # Save the array to a binary file
        np.save('ForceVelocityInverseData.npy', [y0,y1,y2])
        file.close()

#CurveForceVelocityInverse.test()