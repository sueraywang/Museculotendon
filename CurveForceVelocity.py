import numpy as np
from CurveBase import CurveBase
from SmoothSegmentedFunction import SmoothSegmentedFunction

class CurveForceVelocity(CurveBase):
    # CurveForceVelocity
    # Doxygen API
    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1ForceVelocityCurve.html
    # Source
    # https://github.com/opensim-org/opensim-core/blob/master/OpenSim/Actuators/ForceVelocityCurve.h

    def __init__(self, concentric_slope_at_vmax=None, concentric_slope_near_vmax=None,
                 isometric_slope=None, eccentric_slope_at_vmax=None,
                 eccentric_slope_near_vmax=None, max_eccentric_velocity_force_multiplier=None,
                 concentric_curviness=None, eccentric_curviness=None):
        """
        This class serves as a serializable ForceVelocityCurve for use in muscle
        models. The force-velocity curve is dimensionless: force is normalized to
        maximum isometric force and velocity is normalized to the maximum muscle
        contraction velocity (vmax), where vmax is expressed in units of
        optimal_fiber_lengths per second. Negative normalized velocities correspond
        to concentric contraction (i.e., shortening). The force-velocity curve is
        constructed from 8 properties:
        """

        if concentric_slope_at_vmax is None and concentric_slope_near_vmax is None and \
           isometric_slope is None and eccentric_slope_at_vmax is None and \
           eccentric_slope_near_vmax is None and max_eccentric_velocity_force_multiplier is None and \
           concentric_curviness is None and eccentric_curviness is None:
            # Default parameters
            concentric_slope_at_vmax = 0.0
            concentric_slope_near_vmax = 0.25
            isometric_slope = 5.0
            eccentric_slope_at_vmax = 0.0
            eccentric_slope_near_vmax = 0.15  # guess for now
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

        # From ForceVelocityCurve::buildCurve
        self.m_curve = SmoothSegmentedFunction.createFiberForceVelocityCurve(
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

    def calc_value(self, norm_fiber_velocity):
        """
        Evaluates the force-velocity curve at a normalized fiber velocity of
        'norm_fiber_velocity'.
        """
        return self.m_curve.calc_value(norm_fiber_velocity)

    def calc_derivative(self, norm_fiber_velocity, order):
        """
        Calculates the derivative of the force-velocity multiplier with respect
        to the normalized fiber velocity.

        :param norm_fiber_velocity: The normalized velocity of the muscle fiber.
        :param order: The order of the derivative. Only values of 0, 1, and 2 are acceptable.

        :return: The derivative of the force-velocity curve with respect to the
        normalized fiber velocity.
        """
        if not (0 <= order <= 2):
            raise ValueError(f"order must be 0, 1, or 2, but {order} was entered")
        return self.m_curve.calc_derivative(norm_fiber_velocity, order)

    def calc_val_deriv(self, norm_fiber_velocity):
        """
        Calculates the value and derivatives of the force-velocity curve at a normalized fiber velocity.
        """
        return self.m_curve.calc_val_deriv(norm_fiber_velocity)

    @staticmethod
    def test():
        curve = CurveForceVelocity()
        cname = 'Force Velocity Curve'
        x = np.linspace(-1.5, 1.5, 300)
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y0, y1, y2 = CurveBase.test(curve, cname, x)
        # Opening a file
        file = open('ForceVelocityData.npy', 'w')
        # Save the array to a binary file
        np.save('ForceVelocityData.npy', [y0,y1,y2])
        file.close()

#CurveForceVelocity.test()