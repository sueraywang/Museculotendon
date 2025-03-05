import numpy as np
from CurveBase import CurveBase
from SmoothSegmentedFunction import SmoothSegmentedFunction

class CurveTendonForceLength(CurveBase):
    """
    CurveTendonForceLength
    Doxygen API: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1TendonForceLengthCurve.html
    Source: https://github.com/opensim-org/opensim-core/blob/master/OpenSim/Actuators/TendonForceLengthCurve.h
    """

    def __init__(self, strain_at_one_norm_force=None, stiffness_at_one_norm_force=None,
                 norm_force_at_toe_end=None, curviness=None):
        """
        This class serves as a serializable TendonForceLengthCurve for use in muscle
        models. The tendon-force-length curve is dimensionless: force is normalized
        to maximum isometric force and length is normalized to tendon slack length.
        The user can adjust the strain the tendon undergoes at 1 unit load (e0), its
        stiffness at a strain of e0, and the shape of the tendon curve (its
        'curviness').
        """

        if strain_at_one_norm_force is None:
            # Default parameters
            strain_at_one_norm_force = 0.049  # 4.9% strain matches in-vivo measurements
            
            # From TendonForceLengthCurve::ensureCurveUpToDate()
            e0 = strain_at_one_norm_force

            # Assign the stiffness. By eyeball, this agrees well with various in-vitro tendon data.
            stiffness_at_one_norm_force = 1.375 / e0
            norm_force_at_toe_end = 2.0 / 3.0
            curviness = 0.5

        elif all(v is not None for v in [strain_at_one_norm_force, stiffness_at_one_norm_force,
                                         norm_force_at_toe_end, curviness]):
            pass
        else:
            raise ValueError("Please provide 0, 1, or 4 arguments")

        self.strain_at_one_norm_force = strain_at_one_norm_force
        self.stiffness_at_one_norm_force = stiffness_at_one_norm_force
        self.norm_force_at_toe_end = norm_force_at_toe_end
        self.curviness = curviness

        # From TendonForceLengthCurve::buildCurve
        self.m_curve = SmoothSegmentedFunction.createTendonForceLengthCurve(
            self.strain_at_one_norm_force,
            self.stiffness_at_one_norm_force,
            self.norm_force_at_toe_end,
            self.curviness,
            False
        )

    @staticmethod
    def test():
        curve = CurveTendonForceLength()
        cname = 'Tendon Force Length Curve'
        x = np.linspace(0.99, 1.05, 500)
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y0, y1, y2 = CurveBase.test(curve, cname, x)
        # Opening a file
        file = open('TendonForceLengthData.npy', 'w')
        # Save the array to a binary file
        np.save('TendonForceLengthData.npy', [y0,y1,y2])
        file.close()

#CurveTendonForceLength.test()