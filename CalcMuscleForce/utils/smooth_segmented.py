from CalcMuscleForce.utils.quintic_bezier import *

class SmoothSegmentedFunction:
    """Implementation of smooth segmented functions using Bezier curves"""
    
    def __init__(
        self,
        m_x: List[List[float]],
        m_y: List[List[float]],
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        dydx0: float,
        dydx1: float,
        compute_integral: bool = False,
        intx0x1: bool = True
    ):
        """
        Initialize smooth segmented function
        
        Args:
            m_x: x coordinates of control points for each section
            m_y: y coordinates of control points for each section
            x0: Domain minimum
            x1: Domain maximum
            y0: Value at x0
            y1: Value at x1
            dydx0: Derivative at x0
            dydx1: Derivative at x1
            compute_integral: Whether to compute integral (not supported yet)
            intx0x1: Whether to calculate integral from x0 to x1
        """
        assert not compute_integral, "Integration not yet supported"
        
        self.m_x_vec = m_x
        self.m_y_vec = m_y
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.dydx0 = dydx0
        self.dydx1 = dydx1
        self.compute_integral = compute_integral
        self.intx0x1 = intx0x1
        self.num_bezier_sections = len(m_x)
    
    def get_domains(self) -> List[float]:
        """Get the domain boundaries"""
        domains = []
        for i in range(self.num_bezier_sections):
            domains.append(self.m_x_vec[i][0])
        domains.append(self.m_x_vec[self.num_bezier_sections - 1][5])
        return domains
    
    def calc_value(self, x: float) -> float:
        """Calculate the value at a given point"""
        if self.x0 <= x <= self.x1:
            # Find the Bezier section that contains x
            idx = -1
            for i in range(self.num_bezier_sections):
                if self.m_x_vec[i][0] <= x <= self.m_x_vec[i][5]:
                    idx = i
                    break
            
            if idx == -1:
                # Handle edge case: x is exactly at x1
                if abs(x - self.x1) < np.finfo(float).eps:
                    idx = self.num_bezier_sections - 1
                else:
                    raise RuntimeError("Value not within Bezier curve range")
            
            # Extract x control points for this section
            x_pts = [self.m_x_vec[idx][i] for i in range(6)]
            
            # Compute parameter u
            u = QuinticBezierCurve.calc_u(x, x_pts)
            
            # Create points array for calcVal
            pts = [[0.0, self.m_y_vec[idx][i]] for i in range(6)]
            
            # Calculate the value
            y_val = QuinticBezierCurve.calc_val(u, pts)
        else:
            # Outside the curve range, use linear extrapolation
            if x < self.x0:
                y_val = self.y0 + self.dydx0 * (x - self.x0)
            else:
                y_val = self.y1 + self.dydx1 * (x - self.x1)
        
        return y_val
    
    def calc_derivative(self, x: float, order: int) -> float:
        """Calculate derivative of specified order"""
        if order == 0:
            return self.calc_value(x)
        
        if self.x0 <= x <= self.x1:
            # Find the Bezier section that contains x
            idx = -1
            for i in range(self.num_bezier_sections):
                if self.m_x_vec[i][0] <= x <= self.m_x_vec[i][5]:
                    idx = i
                    break
            
            if idx == -1:
                # Handle edge case: x is exactly at x1
                if abs(x - self.x1) < np.finfo(float).eps:
                    idx = self.num_bezier_sections - 1
                else:
                    raise RuntimeError("Value not within Bezier curve range")
            
            # Extract x and y control points for this section
            x_pts = [self.m_x_vec[idx][i] for i in range(6)]
            y_pts = [self.m_y_vec[idx][i] for i in range(6)]
            
            # Compute parameter u
            u = QuinticBezierCurve.calc_u(x, x_pts)
            
            # Calculate the derivative
            y_val = QuinticBezierCurve.calc_deriv_dydx(u, x_pts, y_pts, order)
        else:
            # Outside the curve range
            if order == 1:
                y_val = self.dydx0 if x < self.x0 else self.dydx1
            else:
                y_val = 0.0
        
        return y_val
    
    def calc_val_deriv(self, x: float) -> List[float]:
        """Calculate value and derivatives (up to 2nd) at once"""
        result = [0.0, 0.0, 0.0]  # y0, y1, y2
        
        if self.x0 <= x <= self.x1:
            # Find the Bezier section that contains x
            idx = -1
            for i in range(self.num_bezier_sections):
                if self.m_x_vec[i][0] <= x <= self.m_x_vec[i][5]:
                    idx = i
                    break
            
            if idx == -1:
                # Handle edge case: x is exactly at x1
                if abs(x - self.x1) < np.finfo(float).eps:
                    idx = self.num_bezier_sections - 1
                else:
                    raise RuntimeError("Value not within Bezier curve range")
            
            # Extract x and y control points for this section
            x_pts = [self.m_x_vec[idx][i] for i in range(6)]
            y_pts = [self.m_y_vec[idx][i] for i in range(6)]
            
            # Compute parameter u
            u = QuinticBezierCurve.calc_u(x, x_pts)
            
            # Create points array for calcVal
            pts = [[0.0, y_pts[i]] for i in range(6)]
            
            # Calculate the value and derivatives
            result[0] = QuinticBezierCurve.calc_val(u, pts)
            result[1] = QuinticBezierCurve.calc_deriv_dydx(u, x_pts, y_pts, 1)
            result[2] = QuinticBezierCurve.calc_deriv_dydx(u, x_pts, y_pts, 2)
        else:
            # Outside the curve range, use linear extrapolation
            if x < self.x0:
                result[0] = self.y0 + self.dydx0 * (x - self.x0)
                result[1] = self.dydx0
            else:
                result[0] = self.y1 + self.dydx1 * (x - self.x1)
                result[1] = self.dydx1
            result[2] = 0.0
        
        return result
    
    @staticmethod
    def scale_curviness(curviness: float) -> float:
        """Scale curviness parameter"""
        return 0.1 + 0.8 * curviness
    
    @staticmethod
    def create_fiber_force_length_curve(
        e_zero: float,
        e_iso: float,
        k_low: float,
        k_iso: float,
        curviness: float,
        compute_integral: bool = False
    ) -> 'SmoothSegmentedFunction':
        """
        Create a fiber force-length curve
        
        Args:
            e_zero: Strain at zero force
            e_iso: Strain at one normalized force
            k_low: Stiffness at low force
            k_iso: Stiffness at one normalized force
            curviness: Curviness parameter (0-1)
            compute_integral: Whether to compute integral
            
        Returns:
            SmoothSegmentedFunction object
        """
        assert e_iso > e_zero, "e_iso must be greater than e_zero"
        assert k_iso > 1.0 / (e_iso - e_zero), "k_iso must be greater than 1/(e_iso-e_zero)"
        assert 0 < k_low < 1 / (e_iso - e_zero), "k_low must be between 0 and 1/(e_iso-e_zero)"
        assert 0 <= curviness <= 1, "curviness must be between 0 and 1"
        
        c = SmoothSegmentedFunction.scale_curviness(curviness)
        x_zero = 1 + e_zero
        y_zero = 0
        
        x_iso = 1 + e_iso
        y_iso = 1
        
        delta_x = min(0.1 * (1.0 / k_iso), 0.1 * (x_iso - x_zero))
        
        x_low = x_zero + delta_x
        x_foot = x_zero + 0.5 * (x_low - x_zero)
        y_foot = 0
        y_low = y_foot + k_low * (x_low - x_foot)
        
        # Calculate control points
        p0 = QuinticBezierCurve.calc_corner_control_points(x_zero, y_zero, 0, x_low, y_low, k_low, c)
        p1 = QuinticBezierCurve.calc_corner_control_points(x_low, y_low, k_low, x_iso, y_iso, k_iso, c)
        
        # Create mX and mY arrays
        m_x = [[0.0 for _ in range(6)] for _ in range(2)]
        m_y = [[0.0 for _ in range(6)] for _ in range(2)]
        
        for i in range(6):
            m_x[0][i] = p0[i][0]
            m_y[0][i] = p0[i][1]
            
            m_x[1][i] = p1[i][0]
            m_y[1][i] = p1[i][1]
        
        return SmoothSegmentedFunction(
            m_x, m_y, x_zero, x_iso, y_zero, y_iso, 0.0, k_iso, compute_integral, True
        )
    
    @staticmethod
    def create_tendon_force_length_curve(
        e_iso: float,
        k_iso: float,
        f_toe: float,
        curviness: float,
        compute_integral: bool = False
    ) -> 'SmoothSegmentedFunction':
        """
        Create a tendon force-length curve
        
        Args:
            e_iso: Strain at one normalized force
            k_iso: Stiffness at one normalized force
            f_toe: Normalized force at toe end
            curviness: Curviness parameter (0-1)
            compute_integral: Whether to compute integral
            
        Returns:
            SmoothSegmentedFunction object
        """
        # Check the input arguments
        assert e_iso > 0, "e_iso must be greater than 0"
        assert 0 < f_toe < 1, "f_toe must be greater than 0 and less than 1"
        assert k_iso > (1 / e_iso), "k_iso must be greater than 1/e_iso"
        assert 0 <= curviness <= 1, "curviness must be between 0.0 and 1.0"
        
        # Translate the user parameters to quintic Bezier points
        c = SmoothSegmentedFunction.scale_curviness(curviness)
        x0 = 1.0
        y0 = 0
        dydx0 = 0
        x_iso = 1.0 + e_iso
        y_iso = 1
        dydx_iso = k_iso
        
        # Location where the curved section becomes linear
        y_toe = f_toe
        x_toe = (y_toe - 1) / k_iso + x_iso
        
        # To limit the 2nd derivative of the toe region the line it tends to
        # has to intersect the x axis to the right of the origin
        x_foot = 1.0 + (x_toe - 1.0) / 10.0
        y_foot = 0
        
        # Compute the location of the corner formed by the average slope of the
        # toe and the slope of the linear section
        y_toe_mid = y_toe * 0.5
        x_toe_mid = (y_toe_mid - y_iso) / k_iso + x_iso
        dydx_toe_mid = (y_toe_mid - y_foot) / (x_toe_mid - x_foot)
        
        # Compute the location of the control point to the left of the corner
        x_toe_ctrl = x_foot + 0.5 * (x_toe_mid - x_foot)
        y_toe_ctrl = y_foot + dydx_toe_mid * (x_toe_ctrl - x_foot)
        
        # Compute the Quintic Bezier control points
        p0 = QuinticBezierCurve.calc_corner_control_points(x0, y0, dydx0, x_toe_ctrl, y_toe_ctrl, dydx_toe_mid, c)
        p1 = QuinticBezierCurve.calc_corner_control_points(x_toe_ctrl, y_toe_ctrl, dydx_toe_mid, x_toe, y_toe, dydx_iso, c)
        
        m_x = [[0.0 for _ in range(6)] for _ in range(2)]
        m_y = [[0.0 for _ in range(6)] for _ in range(2)]
        
        for i in range(6):
            m_x[0][i] = p0[i][0]
            m_y[0][i] = p0[i][1]
            
            m_x[1][i] = p1[i][0]
            m_y[1][i] = p1[i][1]
        
        # Instantiate a muscle curve object
        return SmoothSegmentedFunction(
            m_x, m_y,
            x0, x_toe,
            y0, y_toe,
            dydx0, dydx_iso,
            compute_integral,
            True
        )
    
    @staticmethod
    def create_fiber_force_velocity_curve(
        fmax_e: float,
        dydx_c: float,
        dydx_near_c: float,
        dydx_iso: float,
        dydx_e: float,
        dydx_near_e: float,
        conc_curviness: float,
        ecc_curviness: float,
        compute_integral: bool = False
    ) -> 'SmoothSegmentedFunction':
        """
        Create a fiber force-velocity curve
        
        Args:
            fmax_e: Maximum eccentric force
            dydx_c: Derivative at maximum concentric velocity
            dydx_near_c: Derivative near maximum concentric velocity
            dydx_iso: Derivative at isometric velocity
            dydx_e: Derivative at maximum eccentric velocity
            dydx_near_e: Derivative near maximum eccentric velocity
            conc_curviness: Curviness parameter for concentric portion (0-1)
            ecc_curviness: Curviness parameter for eccentric portion (0-1)
            compute_integral: Whether to compute integral
            
        Returns:
            SmoothSegmentedFunction object
        """
        # Ensure that the inputs are within a valid range
        assert fmax_e > 1.0, "fmax_e must be greater than 1"
        assert 0.0 <= dydx_c < 1, "dydx_c must be greater than or equal to 0 and less than 1"
        assert dydx_near_c > dydx_c and dydx_near_c <= 1, "dydx_near_c must be greater than dydx_c and less than or equal to 1"
        assert dydx_iso > 1, "dydx_iso must be greater than (fmax_e-1)/1"
        assert 0.0 <= dydx_e < (fmax_e - 1), "dydx_e must be greater than or equal to 0 and less than fmax_e-1"
        assert dydx_near_e >= dydx_e and dydx_near_e < (fmax_e - 1), "dydx_near_e must be greater than or equal to dydx_e and less than fmax_e-1"
        assert 0 <= conc_curviness <= 1, "conc_curviness must be between 0 and 1"
        assert 0 <= ecc_curviness <= 1, "ecc_curviness must be between 0 and 1"
        
        # Translate the users parameters into Bezier point locations
        c_c = SmoothSegmentedFunction.scale_curviness(conc_curviness)
        c_e = SmoothSegmentedFunction.scale_curviness(ecc_curviness)
        
        # Compute the concentric control point locations
        x_c = -1
        y_c = 0
        x_near_c = -0.9
        y_near_c = y_c + 0.5 * dydx_near_c * (x_near_c - x_c) + 0.5 * dydx_c * (x_near_c - x_c)
        
        x_iso = 0
        y_iso = 1
        
        x_e = 1
        y_e = fmax_e
        x_near_e = 0.9
        y_near_e = y_e + 0.5 * dydx_near_e * (x_near_e - x_e) + 0.5 * dydx_e * (x_near_e - x_e)
        
        conc_pts1 = QuinticBezierCurve.calc_corner_control_points(x_c, y_c, dydx_c, x_near_c, y_near_c, dydx_near_c, c_c)
        conc_pts2 = QuinticBezierCurve.calc_corner_control_points(x_near_c, y_near_c, dydx_near_c, x_iso, y_iso, dydx_iso, c_c)
        ecc_pts1 = QuinticBezierCurve.calc_corner_control_points(x_iso, y_iso, dydx_iso, x_near_e, y_near_e, dydx_near_e, c_e)
        ecc_pts2 = QuinticBezierCurve.calc_corner_control_points(x_near_e, y_near_e, dydx_near_e, x_e, y_e, dydx_e, c_e)
        
        m_x = [[0.0 for _ in range(6)] for _ in range(4)]
        m_y = [[0.0 for _ in range(6)] for _ in range(4)]
        
        for i in range(6):
            m_x[0][i] = conc_pts1[i][0]
            m_x[1][i] = conc_pts2[i][0]
            m_x[2][i] = ecc_pts1[i][0]
            m_x[3][i] = ecc_pts2[i][0]
            
            m_y[0][i] = conc_pts1[i][1]
            m_y[1][i] = conc_pts2[i][1]
            m_y[2][i] = ecc_pts1[i][1]
            m_y[3][i] = ecc_pts2[i][1]
        
        return SmoothSegmentedFunction(
            m_x, m_y,
            x_c, x_e,
            y_c, y_e,
            dydx_c, dydx_e,
            compute_integral,
            True
        )
    
    @staticmethod
    def create_fiber_force_velocity_inverse_curve(
        fmax_e: float,
        dydx_c: float,
        dydx_near_c: float,
        dydx_iso: float,
        dydx_e: float,
        dydx_near_e: float,
        conc_curviness: float,
        ecc_curviness: float,
        compute_integral: bool = False
    ) -> 'SmoothSegmentedFunction':
        """
        Create a fiber force-velocity inverse curve
        
        Args:
            fmax_e: Maximum eccentric force
            dydx_c: Derivative at maximum concentric velocity
            dydx_near_c: Derivative near maximum concentric velocity
            dydx_iso: Derivative at isometric velocity
            dydx_e: Derivative at maximum eccentric velocity
            dydx_near_e: Derivative near maximum eccentric velocity
            conc_curviness: Curviness parameter for concentric portion (0-1)
            ecc_curviness: Curviness parameter for eccentric portion (0-1)
            compute_integral: Whether to compute integral
            
        Returns:
            SmoothSegmentedFunction object
        """
        # Ensure that the inputs are within a valid range
        root_eps = np.sqrt(np.finfo(float).eps)
        assert fmax_e > 1.0, "fmax_e must be greater than 1"
        assert root_eps < dydx_c < 1, "dydx_c must be greater than 0 and less than 1"
        assert dydx_near_c > dydx_c and dydx_near_c < 1, "dydx_near_c must be greater than 0 and less than 1"
        assert dydx_iso > 1, "dydx_iso must be greater than or equal to 1"
        assert root_eps < dydx_e < (fmax_e - 1), "dydx_e must be greater than or equal to 0 and less than fmax_e-1"
        assert dydx_near_e >= dydx_e and dydx_near_e < (fmax_e - 1), "dydx_near_e must be greater than or equal to dydx_e and less than fmax_e-1"
        assert 0 <= conc_curviness <= 1, "conc_curviness must be between 0 and 1"
        assert 0 <= ecc_curviness <= 1, "ecc_curviness must be between 0 and 1"
        
        # Translate the users parameters into Bezier point locations
        c_c = SmoothSegmentedFunction.scale_curviness(conc_curviness)
        c_e = SmoothSegmentedFunction.scale_curviness(ecc_curviness)
        
        # Compute the concentric control point locations
        x_c = -1
        y_c = 0
        x_near_c = -0.9
        y_near_c = y_c + 0.5 * dydx_near_c * (x_near_c - x_c) + 0.5 * dydx_c * (x_near_c - x_c)
        
        x_iso = 0
        y_iso = 1
        
        x_e = 1
        y_e = fmax_e
        x_near_e = 0.9
        y_near_e = y_e + 0.5 * dydx_near_e * (x_near_e - x_e) + 0.5 * dydx_e * (x_near_e - x_e)
        
        conc_pts1 = QuinticBezierCurve.calc_corner_control_points(x_c, y_c, dydx_c, x_near_c, y_near_c, dydx_near_c, c_c)
        conc_pts2 = QuinticBezierCurve.calc_corner_control_points(x_near_c, y_near_c, dydx_near_c, x_iso, y_iso, dydx_iso, c_c)
        ecc_pts1 = QuinticBezierCurve.calc_corner_control_points(x_iso, y_iso, dydx_iso, x_near_e, y_near_e, dydx_near_e, c_e)
        ecc_pts2 = QuinticBezierCurve.calc_corner_control_points(x_near_e, y_near_e, dydx_near_e, x_e, y_e, dydx_e, c_e)
        
        m_x = [[0.0 for _ in range(6)] for _ in range(4)]
        m_y = [[0.0 for _ in range(6)] for _ in range(4)]
        
        for i in range(6):
            # Note: We're swapping X and Y for the inverse curve
            m_y[0][i] = conc_pts1[i][0]
            m_y[1][i] = conc_pts2[i][0]
            m_y[2][i] = ecc_pts1[i][0]
            m_y[3][i] = ecc_pts2[i][0]
            
            m_x[0][i] = conc_pts1[i][1]
            m_x[1][i] = conc_pts2[i][1]
            m_x[2][i] = ecc_pts1[i][1]
            m_x[3][i] = ecc_pts2[i][1]
        
        return SmoothSegmentedFunction(
            m_x, m_y,
            y_c, y_e,
            x_c, x_e,
            1 / dydx_c, 1 / dydx_e,
            compute_integral,
            True
        )
    
    @staticmethod
    def create_fiber_active_force_length_curve(
        x0: float,
        x1: float,
        x2: float,
        x3: float,
        y_low: float,
        dydx: float,
        curviness: float,
        compute_integral: bool = False
    ) -> 'SmoothSegmentedFunction':
        """
        Create a fiber active force-length curve
        
        Args:
            x0: Minimum active normalized fiber length
            x1: Transition normalized fiber length
            x2: Optimal normalized fiber length (typically 1.0)
            x3: Maximum active normalized fiber length
            y_low: Minimum value
            dydx: Slope of the shallow ascending region
            curviness: Curviness parameter (0-1)
            compute_integral: Whether to compute integral
            
        Returns:
            SmoothSegmentedFunction object
        """
        # Ensure that the inputs are within a valid range
        root_eps = np.sqrt(np.finfo(float).eps)
        assert x0 >= 0 and x1 > x0 + root_eps and x2 > x1 + root_eps and x3 > x2 + root_eps, "This must be true: 0 < lce0 < lce1 < lce2 < lce3"
        assert y_low >= 0, "shoulderVal must be greater than, or equal to 0"
        dydx_upper_bound = (1 - y_low) / (x2 - x1)
        assert 0 <= dydx < dydx_upper_bound, "plateauSlope must be greater than 0 and less than the upper bound"
        assert 0 <= curviness <= 1, "curviness must be between 0 and 1"
        
        # Translate the users parameters into Bezier curves
        c = SmoothSegmentedFunction.scale_curviness(curviness)
        
        # The active force length curve is made up of 5 elbow shaped sections.
        # Compute the locations of the joining point of each elbow section.
        
        # Calculate the location of the shoulder
        x_delta = 0.05 * x2  # half the width of the sarcomere
        
        xs = (x2 - x_delta)  # x1 + 0.75*(x2-x1)
        
        # Calculate the intermediate points located on the ascending limb
        y0 = 0
        dydx0 = 0
        y1 = 1 - dydx * (xs - x1)
        dydx01 = 1.25 * (y1 - y0) / (x1 - x0)
        
        x01 = x0 + 0.5 * (x1 - x0)
        y01 = y0 + 0.5 * (y1 - y0)
        
        # Calculate the intermediate points of the shallow ascending plateau
        x1s = x1 + 0.5 * (xs - x1)
        y1s = y1 + 0.5 * (1 - y1)
        dydx1s = dydx
        
        # x2 entered
        y2 = 1
        dydx2 = 0
        
        # Descending limb
        # x3 entered
        y3 = 0
        dydx3 = 0
        
        x23 = (x2 + x_delta) + 0.5 * (x3 - (x2 + x_delta))
        y23 = y2 + 0.5 * (y3 - y2)
        dydx23 = (y3 - y2) / ((x3 - x_delta) - (x2 + x_delta))
        
        # Compute the locations of the control points
        p0 = QuinticBezierCurve.calc_corner_control_points(x0, y_low, dydx0, x01, y01, dydx01, c)
        p1 = QuinticBezierCurve.calc_corner_control_points(x01, y01, dydx01, x1s, y1s, dydx1s, c)
        p2 = QuinticBezierCurve.calc_corner_control_points(x1s, y1s, dydx1s, x2, y2, dydx2, c)
        p3 = QuinticBezierCurve.calc_corner_control_points(x2, y2, dydx2, x23, y23, dydx23, c)
        p4 = QuinticBezierCurve.calc_corner_control_points(x23, y23, dydx23, x3, y_low, dydx3, c)
        
        m_x = [[0.0 for _ in range(6)] for _ in range(5)]
        m_y = [[0.0 for _ in range(6)] for _ in range(5)]
        
        for i in range(6):
            m_x[0][i] = p0[i][0]
            m_x[1][i] = p1[i][0]
            m_x[2][i] = p2[i][0]
            m_x[3][i] = p3[i][0]
            m_x[4][i] = p4[i][0]
            
            m_y[0][i] = p0[i][1]
            m_y[1][i] = p1[i][1]
            m_y[2][i] = p2[i][1]
            m_y[3][i] = p3[i][1]
            m_y[4][i] = p4[i][1]
        
        return SmoothSegmentedFunction(
            m_x, m_y,
            x0, x3,
            y_low, y_low,
            0, 0,
            compute_integral,
            True
        )