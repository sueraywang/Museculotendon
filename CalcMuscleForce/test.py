"""
Muscle Force Calculation - Python Implementation

This project provides a Python implementation of a muscle force calculation model
translated from C++ code. The model includes various curves for representing muscle
properties like force-length and force-velocity relationships.

Main components:
- CurveBase: Base class for all muscle curve types
- Various curve implementations (ActiveForceLength, FiberForceLength, etc.)
- QuinticBezierCurve: Implementation of quintic Bezier curves for smooth interpolation
- SmoothSegmentedFunction: Implementation of smooth segmented functions using Bezier curves
- NewtonSolver: Newton-Raphson solver for finding roots of functions
- Muscle: Main muscle model class that composes all components
"""

import numpy as np
from typing import List, Tuple, Callable
import math
import time


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


class NewtonSolver:
    """Newton-Raphson solver for f(x) = 0"""
    
    @staticmethod
    def solve(
        func: Callable[[float], Tuple[float, float]],
        x0: float,
        tol: float = 1e-9,
        kmax: int = 100,
        dxmax: float = float('inf')
    ) -> float:
        """
        Solve f(x) = 0 using Newton's method
        
        Args:
            func: Function that returns (f(x), f'(x))
            x0: Initial guess
            tol: Tolerance for convergence
            kmax: Maximum number of iterations
            dxmax: Maximum step size
            
        Returns:
            Solution x where f(x) ≈ 0
        """
        x = x0
        k = 1
        
        while k < kmax:
            # Evaluate function and its Jacobian
            f, J = func(x)
            
            # Check for convergence in function value
            if abs(f) < tol:
                break
            
            # Calculate step
            dx = -f / J
            
            # Check for step size limit
            if abs(dx) > dxmax:
                dx = dxmax if dx > 0 else -dxmax
            
            # Update x
            x = x + dx
            
            # Check for convergence in step size
            if abs(dx) < tol:
                break
            
            k += 1
        
        return x


class QuinticBezierCurve:
    """Implementation of quintic Bezier curves for smooth interpolation"""
    
    @staticmethod
    def calc_corner_control_points(
        x0: float, y0: float, dydx0: float,
        x1: float, y1: float, dydx1: float,
        curviness: float
    ) -> List[List[float]]:
        """
        Calculate control points for a corner (elbow) in the curve
        
        Args:
            x0, y0: First point coordinates
            dydx0: Derivative at first point
            x1, y1: Second point coordinates
            dydx1: Derivative at second point
            curviness: Curviness parameter (0-1)
            
        Returns:
            List of control points
        """
        root_eps = np.sqrt(np.finfo(float).eps)
        
        # Find the intersection of the two lines
        if abs(dydx0 - dydx1) > root_eps:
            xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1)
        else:
            xC = (x1 + x0) / 2
        
        yC = (xC - x1) * dydx1 + y1
        
        # Create the six Bezier control points
        xy_pts = [[0.0, 0.0] for _ in range(6)]
        
        xy_pts[0] = [x0, y0]
        xy_pts[5] = [x1, y1]
        
        xy_pts[1] = [x0 + curviness * (xC - xy_pts[0][0]), y0 + curviness * (yC - xy_pts[0][1])]
        xy_pts[2] = xy_pts[1][:]
        
        xy_pts[3] = [xy_pts[5][0] + curviness * (xC - xy_pts[5][0]), xy_pts[5][1] + curviness * (yC - xy_pts[5][1])]
        xy_pts[4] = xy_pts[3][:]
        
        return xy_pts
    
    @staticmethod
    def calc_val(u: float, pts: List[List[float]]) -> float:
        """
        Calculate the value of the curve at parameter u
        
        Args:
            u: Parameter value (0-1)
            pts: Control points
            
        Returns:
            Value of the curve at u
        """
        u5 = 1
        u4 = u
        u3 = u4 * u
        u2 = u3 * u
        u1 = u2 * u
        u0 = u1 * u
        
        t2 = u1 * 5
        t3 = u2 * 10
        t4 = u3 * 10
        t5 = u4 * 5
        t9 = u0 * 5
        t10 = u1 * 20
        t11 = u2 * 30
        t15 = u0 * 10
        
        val = (
            pts[0][1] * (u0 * -1 + t2 - t3 + t4 - t5 + u5 * 1) +
            pts[1][1] * (t9 - t10 + t11 + u3 * -20 + t5) +
            pts[2][1] * (-t15 + u1 * 30 - t11 + t4) +
            pts[3][1] * (t15 - t10 + t3) +
            pts[4][1] * (-t9 + t2) +
            pts[5][1] * u0 * 1
        )
        
        return val
    
    @staticmethod
    def calc_deriv_u(u: float, pts: List[List[float]], order: int) -> float:
        """
        Calculate derivatives with respect to u parameter
        
        Args:
            u: Parameter value (0-1)
            pts: Control points
            order: Order of derivative (1-5)
            
        Returns:
            Derivative of order 'order' at u
        """
        p0 = pts[0][1]
        p1 = pts[1][1]
        p2 = pts[2][1]
        p3 = pts[3][1]
        p4 = pts[4][1]
        p5 = pts[5][1]
        val = 0.0
        
        if order == 1:
            t1 = u * u  # u ^ 2
            t2 = t1 * t1  # t1 ^ 2
            t4 = t1 * u
            t5 = t4 * 20.0
            t6 = t1 * 30.0
            t7 = u * 20.0
            t10 = t2 * 25.0
            t11 = t4 * 80.0
            t12 = t1 * 90.0
            t16 = t2 * 50.0
            val = p0 * (t2 * (-5.0) + t5 - t6 + t7 - 5.0) + \
                p1 * (t10 - t11 + t12 + u * (-40.0) + 5.0) + \
                p2 * (-t16 + t4 * 120.0 - t12 + t7) + \
                p3 * (t16 - t11 + t6) + \
                p4 * (-t10 + t5) + \
                p5 * t2 * 5.0
        elif order == 2:
            t1 = u * u  # u ^ 2
            t2 = t1 * u
            t4 = t1 * 60.0
            t5 = u * 60.0
            t8 = t2 * 100.0
            t9 = t1 * 240.0
            t10 = u * 180.0
            t13 = t2 * 200.0
            val = p0 * (t2 * (-20.0) + t4 - t5 + 20.0) + \
                p1 * (t8 - t9 + t10 - 40.0) + \
                p2 * (-t13 + t1 * 360.0 - t10 + 20.0) + \
                p3 * (t13 - t9 + t5) + \
                p4 * (-t8 + t4) + \
                p5 * t2 * 20.0
        elif order == 3:
            t1 = u * u  # u ^ 2
            t3 = u * 120.0
            t6 = t1 * 300.0
            t7 = u * 480.0
            t10 = t1 * 600.0
            val = p0 * (t1 * (-60.0) + t3 - 60.0) + \
                p1 * (t6 - t7 + 180.0) + \
                p2 * (-t10 + u * 720.0 - 180.0) + \
                p3 * (t10 - t7 + 60.0) + \
                p4 * (-t6 + t3) + \
                p5 * t1 * 60.0
        elif order == 4:
            t4 = u * 600.0
            t7 = u * 1200.0
            val = p0 * (u * (-120.0) + 120.0) + \
                p1 * (t4 - 480.0) + \
                p2 * (-t7 + 720.0) + \
                p3 * (t7 - 480.0) + \
                p4 * (-t4 + 120.0) + \
                p5 * u * 120.0
        elif order == 5:
            val = p0 * (-120.0) + \
                p1 * 600.0 + \
                p2 * (-1200.0) + \
                p3 * 1200.0 + \
                p4 * (-600.0) + \
                p5 * 120.0
        else:
            val = 0
        
        return val
    
    @staticmethod
    def calc_deriv_dydx(u: float, x_pts: List[float], y_pts: List[float], order: int) -> float:
        """
        Calculate derivatives dy/dx
        
        Args:
            u: Parameter value (0-1)
            x_pts: x-coordinates of control points
            y_pts: y-coordinates of control points
            order: Order of derivative
            
        Returns:
            Derivative of order 'order' at u
        """
        val = 0.0
        
        # Create points array for x and y separately
        x_pts_array = [[0.0, x] for x in x_pts]
        y_pts_array = [[0.0, y] for y in y_pts]
        
        # Compute the derivative d^n y / dx^n
        if order == 1:  # Calculate dy/dx
            dxdu = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 1)
            dydu = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 1)
            val = dydu / dxdu
        elif order == 2:  # Calculate d^2y/dx^2
            dxdu = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 1)
            dydu = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 1)
            d2xdu2 = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 2)
            d2ydu2 = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 2)
            t1 = 1.0 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            val = (d2ydu2 * t1 - dydu / t3 * d2xdu2) * t1
        elif order == 3:  # Calculate d^3y/dx^3
            dxdu = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 1)
            dydu = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 1)
            d2xdu2 = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 2)
            d2ydu2 = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 2)
            d3xdu3 = QuinticBezierCurve.calc_deriv_u(u, x_pts_array, 3)
            d3ydu3 = QuinticBezierCurve.calc_deriv_u(u, y_pts_array, 3)
            t1 = 1.0 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            t4 = 1.0 / t3
            t11 = d2xdu2 * d2xdu2  # (d2xdu2 ^ 2)
            t14 = dydu * t4
            val = ((d3ydu3 * t1 - 2 * d2ydu2 * t4 * d2xdu2
                + 2 * dydu / t3 / dxdu * t11 - t14 * d3xdu3) * t1
                - (d2ydu2 * t1 - t14 * d2xdu2) * t4 * d2xdu2) * t1
        else:
            # Higher order derivatives are complex and not needed for most applications
            raise RuntimeError("Derivatives of order > 3 not implemented yet")
        
        return val
    
    @staticmethod
    def clamp_u(u: float) -> float:
        """Clamp u to [0,1]"""
        return max(0.0, min(1.0, u))
    
    @staticmethod
    def calc_u(ax: float, bezier_pts_x: List[float]) -> float:
        """
        Compute u parameter for a given x value
        
        Args:
            ax: x value
            bezier_pts_x: x-coordinates of control points
            
        Returns:
            Parameter u corresponding to x
        """
        # Check to make sure that ax is in the curve domain
        min_x = min(bezier_pts_x)
        max_x = max(bezier_pts_x)
        
        if not (min_x <= ax <= max_x):
            raise RuntimeError(
                "Input ax is not in the domain of the Bezier curve specified by the control points"
            )
        
        # Create points array
        pts = [[0.0, x] for x in bezier_pts_x]
        
        # Define the objective function
        def objective(u):
            x = QuinticBezierCurve.calc_val(u, pts)
            dxdu = QuinticBezierCurve.calc_deriv_u(u, pts, 1)
            return x - ax, dxdu
        
        # Initial guess
        u_init = (ax - min_x) / (max_x - min_x)
        
        # Use Newton's method to find the root
        solver = NewtonSolver()
        return solver.solve(objective, u_init, 1e-9, 10)
    
    @staticmethod
    def calc_index(x: float, bezier_pts_x: List[List[float]]) -> int:
        """
        Find the index of the Bezier section containing x
        
        Args:
            x: x value
            bezier_pts_x: x-coordinates of control points for each section
            
        Returns:
            Index of the section containing x
        """
        idx = 0
        flag_found = False
        
        # Iterate over columns in bezier_pts_x
        for i, pts in enumerate(bezier_pts_x):
            if pts[0] <= x < pts[5]:
                idx = i
                flag_found = True
                break
        
        # Check if x is identically the last point
        if not flag_found and x == bezier_pts_x[-1][5]:
            idx = len(bezier_pts_x) - 1
            flag_found = True
        
        # Optional: Raise an error if the value x is not within the Bezier curve set
        if not flag_found:
            raise RuntimeError("A value of x was used that is not within the Bezier curve set")
        
        return idx


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


class CurveForceVelocity(CurveBase):
    """Curve for force-velocity relationship"""
    
    def __init__(
        self,
        concentric_slope_at_vmax: float = 0.0,
        concentric_slope_near_vmax: float = 0.0,
        isometric_slope: float = 0.0,
        eccentric_slope_at_vmax: float = 0.0,
        eccentric_slope_near_vmax: float = 0.0,
        max_eccentric_velocity_force_multiplier: float = 0.0,
        concentric_curviness: float = 0.0,
        eccentric_curviness: float = 0.0
    ):
        """
        Initialize force-velocity curve
        
        Args:
            concentric_slope_at_vmax: Slope at maximum concentric velocity
            concentric_slope_near_vmax: Slope near maximum concentric velocity
            isometric_slope: Slope at isometric velocity
            eccentric_slope_at_vmax: Slope at maximum eccentric velocity
            eccentric_slope_near_vmax: Slope near maximum eccentric velocity
            max_eccentric_velocity_force_multiplier: Force multiplier at maximum eccentric velocity
            concentric_curviness: Curviness parameter for concentric portion (0-1)
            eccentric_curviness: Curviness parameter for eccentric portion (0-1)
        """
        if concentric_slope_at_vmax == 0.0:
            # Default parameters
            self.concentric_slope_at_vmax = 0.0
            self.concentric_slope_near_vmax = 0.25
            self.isometric_slope = 5.0
            self.eccentric_slope_at_vmax = 0.0
            self.eccentric_slope_near_vmax = 0.15  # guess for now
            self.max_eccentric_velocity_force_multiplier = 1.4
            self.concentric_curviness = 0.6
            self.eccentric_curviness = 0.9
        else:
            # Use provided values
            self.concentric_slope_at_vmax = concentric_slope_at_vmax
            self.concentric_slope_near_vmax = concentric_slope_near_vmax
            self.isometric_slope = isometric_slope
            self.eccentric_slope_at_vmax = eccentric_slope_at_vmax
            self.eccentric_slope_near_vmax = eccentric_slope_near_vmax
            self.max_eccentric_velocity_force_multiplier = max_eccentric_velocity_force_multiplier
            self.concentric_curviness = concentric_curviness
            self.eccentric_curviness = eccentric_curviness
        
        # Build the curve
        self.curve = SmoothSegmentedFunction.create_fiber_force_velocity_curve(
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
    
    def calc_value(self, x: float) -> float:
        """Calculate the value of the curve at a given point"""
        return self.curve.calc_value(x)
    
    def calc_derivative(self, x: float, order: int) -> float:
        """Calculate the derivative of the curve at a given point"""
        if order < 0 or order > 2:
            raise ValueError("Order must be 0, 1, or 2")
        return self.curve.calc_derivative(x, order)
    
    def calc_val_deriv(self, x: float) -> List[float]:
        """Calculate both value and derivatives in one call (for performance)"""
        return self.curve.calc_val_deriv(x)


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


class Muscle:
    """Main muscle model class"""
    
    def __init__(
        self,
        l_m_opt: float,
        l_t_slack: float,
        v_m_max: float,
        alpha_m_opt: float,
        f_m_opt: float,
        beta: float = 0.1,
        a_min: float = 0.01,
        tau_a: float = 0.01,
        tau_d: float = 0.4
    ):
        """
        Initialize muscle model
        
        Args:
            l_m_opt: Optimal muscle length
            l_t_slack: Tendon slack length
            v_m_max: Maximum contraction velocity
            alpha_m_opt: Pennation angle at optimal muscle length
            f_m_opt: Peak isometric force
            beta: Damping coefficient
            a_min: Minimum activation
            tau_a: Activation time constant
            tau_d: Deactivation time constant
        """
        self.beta = beta
        self.l_m_opt = l_m_opt
        self.l_t_slack = l_t_slack
        self.v_m_max = v_m_max
        self.alpha_m_opt = alpha_m_opt
        self.f_m_opt = f_m_opt
        self.a_min = a_min
        self.tau_a = tau_a
        self.tau_d = tau_d
        
        # Initialize curves
        self.curve_afl = CurveActiveForceLength()
        self.curve_pfl = CurveFiberForceLength()
        self.curve_tfl = CurveTendonForceLength()
        self.curve_fv = CurveForceVelocity()
        
        # Initialize derived parameters
        self.initialize()
    
    def initialize(self):
        """Initialize derived parameters"""
        self.alpha_max = math.acos(0.1)
        self.h = self.l_m_opt * math.sin(self.alpha_m_opt)
        self.l_mt = self.l_t_slack + self.l_m_opt * math.cos(self.alpha_m_opt)
        
        if self.alpha_max > 1e-6:
            min_pennated_fiber_length = self.h / math.sin(self.alpha_max)
        else:
            min_pennated_fiber_length = self.l_m_opt * 0.01
        
        min_active_fiber_length = self.curve_afl.get_min_norm_active_fiber_length() * self.l_m_opt
        self.l_m_min = max(min_active_fiber_length, min_pennated_fiber_length)
    
    def calc_pennation_angle(self, l_m: float) -> float:
        """
        Calculate pennation angle based on muscle length
        
        Args:
            l_m: Muscle length
            
        Returns:
            Pennation angle in radians
        """
        alpha_m = 0.0
        
        if self.alpha_m_opt > np.finfo(float).eps:
            if l_m > self.l_m_min:
                sin_alpha = self.h / l_m
                if sin_alpha < self.alpha_max:
                    alpha_m = math.asin(sin_alpha)
                else:
                    alpha_m = self.alpha_max
            else:
                alpha_m = self.alpha_max
        
        return alpha_m
    
    def calc_pennation_angle_tilde(self, l_m_tilde: float) -> float:
        """
        Calculate pennation angle based on normalized muscle length
        
        Args:
            l_m_tilde: Normalized muscle length (l_m / l_m_opt)
            
        Returns:
            Pennation angle in radians
        """
        alpha_m = 0.0
        
        if self.alpha_m_opt > np.finfo(float).eps:
            h_tilde = self.h / self.l_m_opt
            if l_m_tilde > self.l_m_min / self.l_m_opt:
                sin_alpha = h_tilde / l_m_tilde
                if sin_alpha < self.alpha_max:
                    alpha_m = math.asin(sin_alpha)
                else:
                    alpha_m = self.alpha_max
            else:
                alpha_m = self.alpha_max
        
        return alpha_m
    
    def force_balance(
        self,
        v_m_tilde: float,
        act: float,
        afl: float,
        pfl: float,
        tfl: float,
        curve_fv: CurveForceVelocity,
        cos_alpha_m: float
    ) -> Tuple[float, float]:
        """
        Force balance function for Newton solver
        
        Args:
            v_m_tilde: Normalized muscle velocity
            act: Activation level
            afl: Active force-length multiplier
            pfl: Passive force-length multiplier
            tfl: Tendon force-length multiplier
            curve_fv: Force-velocity curve
            cos_alpha_m: Cosine of pennation angle
            
        Returns:
            Tuple of (f, J) where f is the force balance equation and J is its derivative
        """
        deriv = curve_fv.calc_val_deriv(v_m_tilde)
        fv = deriv[0]
        dfv = deriv[1]
        
        f_m = act * afl * fv + pfl + self.beta * v_m_tilde
        f = f_m * cos_alpha_m - tfl
        J = (act * afl * dfv + self.beta) * cos_alpha_m
        
        return f, J
    
    def compute_vel_tilde(
        self,
        l_m_tilde: float,
        l_t_tilde: float,
        act: float,
        alpha_m: float
    ) -> float:
        """
        Compute normalized muscle velocity
        
        Args:
            l_m_tilde: Normalized muscle length
            l_t_tilde: Normalized tendon length
            act: Activation level
            alpha_m: Pennation angle
            
        Returns:
            Normalized muscle velocity
        """
        cos_alpha_m = math.cos(alpha_m)
        afl = self.curve_afl.calc_value(l_m_tilde)
        pfl = self.curve_pfl.calc_value(l_m_tilde)
        tfl = self.curve_tfl.calc_value(l_t_tilde)
        v_m_tilde_init = 0.0
        
        # Define the objective function for Newton solver
        def objective(v_m_tilde):
            return self.force_balance(v_m_tilde, act, afl, pfl, tfl, self.curve_fv, cos_alpha_m)
        
        # Use Newton solver to find the muscle fiber velocity
        v_m_tilde = NewtonSolver.solve(objective, v_m_tilde_init)
        
        return v_m_tilde
    
    def compute_vel(self, l_m: float, l_mt: float, act: float, alpha_m: float) -> float:
        """
        Compute muscle velocity
        
        Args:
            l_m: Muscle length
            l_mt: Musculotendon length
            act: Activation level
            alpha_m: Pennation angle
            
        Returns:
            Muscle velocity
        """
        l_m_tilde = l_m / self.l_m_opt
        l_t = l_mt - l_m * math.cos(alpha_m)
        l_t_tilde = l_t / self.l_t_slack
        
        v_m_tilde = self.compute_vel_tilde(l_m_tilde, l_t_tilde, act, alpha_m)
        v_m = v_m_tilde * self.l_m_opt * self.v_m_max
        
        return v_m
    
    def muscle_force(self, v_m: float, l_m: float, act: float, alpha_m: float) -> float:
        """
        Calculate muscle force
        
        Args:
            v_m: Muscle velocity
            l_m: Muscle length
            act: Activation level
            alpha_m: Pennation angle
            
        Returns:
            Muscle force
        """
        l_m_tilde = l_m / self.l_m_opt
        v_m_tilde = v_m / (self.l_m_opt * self.v_m_max)
        
        afl = self.curve_afl.calc_value(l_m_tilde)
        pfl = self.curve_pfl.calc_value(l_m_tilde)
        deriv = self.curve_fv.calc_val_deriv(v_m_tilde)
        fv = deriv[0]
        
        f_m = act * afl * fv + pfl + self.beta * v_m_tilde
        f = f_m * math.cos(alpha_m) * self.f_m_opt
        
        return f
    
    def muscle_force_vectorized(
        self,
        v_m: List[float],
        l_m: List[float],
        act: List[float],
        alpha_m: List[float]
    ) -> List[float]:
        """
        Vectorized muscle force calculation for performance
        
        Args:
            v_m: List of muscle velocities
            l_m: List of muscle lengths
            act: List of activation levels
            alpha_m: List of pennation angles
            
        Returns:
            List of muscle forces
        """
        n = len(v_m)
        forces = [0.0] * n
        
        for i in range(n):
            forces[i] = self.muscle_force(v_m[i], l_m[i], act[i], alpha_m[i])
        
        return forces


def main():
    """Example usage and performance test"""
    # Create muscle with default parameters
    muscle = Muscle(
        l_m_opt=0.1,    # Optimal muscle length
        l_t_slack=0.2,  # Tendon slack length
        v_m_max=10,     # Maximum contraction velocity
        alpha_m_opt=math.pi/6,  # Pennation angle at optimal length
        f_m_opt=1.0     # Peak isometric force
    )
    
    print("Muscle model initialized")
    
    # Input data size
    data_size = 1000
    
    # Create input data vectors
    v_m = [0.0] * data_size
    l_m = [0.0] * data_size
    act = [0.0] * data_size
    alpha_m = [0.0] * data_size
    
    # Fill with sample data
    for i in range(data_size):
        v_m[i] = (i % 20 - 10) * 0.1                     # Velocities between -1 and 1
        l_m[i] = 0.08 + (i % 40) * 0.001                 # Lengths around optimal length
        act[i] = 0.2 + (i % 9) * 0.1                     # Activations between 0.2 and 1.0
        alpha_m[i] = muscle.calc_pennation_angle(l_m[i]) # Pennation angles based on length
    
    # Test individual force computation
    test_force = muscle.muscle_force(v_m[0], l_m[0], act[0], alpha_m[0])
    print(f"Single force test: {test_force}")
    
    # Measure performance of vectorized computation
    start = time.time()
    
    forces = muscle.muscle_force_vectorized(v_m, l_m, act, alpha_m)
    
    end = time.time()
    elapsed = (end - start) * 1000  # Convert to milliseconds
    
    print(f"Computed {data_size} muscle forces in {elapsed:.2f} ms")
    print(f"Average time per force computation: {elapsed / data_size:.6f} ms")
    
    # Output first few values for verification
    print("\nSample output (first 5 values):")
    for i in range(min(5, data_size)):
        print(f"vM={v_m[i]:.3f}, lM={l_m[i]:.3f}, "
              f"act={act[i]:.1f}, alphaM={alpha_m[i]:.3f} "
              f"-> Force={forces[i]:.6f}")


if __name__ == "__main__":
    main()