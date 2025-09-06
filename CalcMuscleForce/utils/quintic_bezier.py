import numpy as np
from typing import List
from CalcMuscleForce.utils.newton_solver import NewtonSolver

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