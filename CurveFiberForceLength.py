import numpy as np
from CurveBase import CurveBase
from SmoothSegmentedFunction import SmoothSegmentedFunction

class CurveFiberForceLength(CurveBase):
    # CurveFiberForceLength
    # Doxygen API
    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1FiberForceLengthCurve.html
    # Source
    # https://github.com/opensim-org/opensim-core/blob/master/OpenSim/Actuators/FiberForceLengthCurve.h

    def __init__(self, strain_at_zero_force=None, strain_at_one_norm_force=None,
                 stiffness_at_low_force=None, stiffness_at_one_norm_force=None,
                 curviness=None):
        """
        This class serves as a serializable FiberForceLengthCurve, commonly used to
        model the parallel elastic element in muscle models. The fiber-force-length
        curve is dimensionless: force is normalized to maximum isometric force and
        length is normalized to resting fiber length. The user can adjust the
        maximum strain at no load and the strain developed under 1 normalized unit
        of force using the fitted curve. Additionally, if desired, it is possible to
        directly set the low-force stiffness of the fiber, the stiffness of the
        fiber at 1 normalized unit of force, and the shape of the curve (its
        'curviness').

        :param strain_at_zero_force: Fiber strain at which the fiber starts to develop force.
        :param strain_at_one_norm_force: Fiber strain at which the fiber develops 1 unit of normalized force.
        :param stiffness_at_low_force: Normalized stiffness when the fiber is just beginning to develop tensile force.
        :param stiffness_at_one_norm_force: Normalized stiffness when the fiber develops a tension of 1 normalized unit of force.
        :param curviness: A dimensionless parameter between 0 and 1 that describes the shape of the curve.
        """
        if strain_at_zero_force is None and strain_at_one_norm_force is None:
            # Default parameters
            strain_at_zero_force = 0.0
            strain_at_one_norm_force = 0.7

            e0 = strain_at_zero_force  # properties of reference curve
            e1 = strain_at_one_norm_force

            # Assign the stiffnesses. These values are based on the Thelen2003
            # default curve and the EDL passive force-length curve found
            # experimentally by Winters, Takahashi, Ward, and Lieber (2011).
            stiffness_at_one_norm_force = 2.0 / (e1 - e0)
            stiffness_at_low_force = 0.2

            # Fit the curviness parameter to the reference curve, which is the one
            # documented by Thelen. This approach is very slow and Thelen's curve
            # was not based on experimental data.
            curviness = 0.75
        elif strain_at_zero_force is not None and strain_at_one_norm_force is not None and \
             stiffness_at_low_force is not None and stiffness_at_one_norm_force is not None and \
             curviness is not None:
            pass
        else:
            raise ValueError("Please provide 0, 2, or 5 arguments")

        self.strain_at_zero_force = strain_at_zero_force
        self.strain_at_one_norm_force = strain_at_one_norm_force
        self.stiffness_at_low_force = stiffness_at_low_force
        self.stiffness_at_one_norm_force = stiffness_at_one_norm_force
        self.curviness = curviness

        # From FiberForceLengthCurve::buildCurve
        self.m_curve = SmoothSegmentedFunction.createFiberForceLengthCurve(
            self.strain_at_zero_force,
            self.strain_at_one_norm_force,
            self.stiffness_at_low_force,
            self.stiffness_at_one_norm_force,
            self.curviness,
            False
        )

    @staticmethod
    def test():
        curve = CurveFiberForceLength()
        cname = 'Passive Force Length Curve'
        x = np.linspace(0.8, 1.75, 1000)
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y0, y1, y2 = CurveBase.test(curve, cname, x)
        # Opening a file
        file = open('PassiveForceLengthData.npy', 'w')
        # Save the array to a binary file
        np.save('PassiveForceLengthData.npy', [y0,y1,y2])
        file.close()

CurveFiberForceLength.test()