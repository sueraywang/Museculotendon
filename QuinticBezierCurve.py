import numpy as np
from fzeroNewtonWithConvergenceCheck import fzero_newton

class QuinticBezierCurve:
    @staticmethod
    def calcCornerControlPoints(x0, y0, dydx0, x1, y1, dydx1, curviness):
        root_eps = np.sqrt(np.finfo(float).eps)
        if abs(dydx0 - dydx1) > root_eps:
            xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1)
        else:
            xC = (x1 + x0) / 2
        yC = (xC - x1) * dydx1 + y1

        xy_pts = np.zeros((6, 2))

        xy_pts[0] = [x0, y0]
        xy_pts[5] = [x1, y1]

        xy_pts[1] = [x0 + curviness * (xC - xy_pts[0, 0]), y0 + curviness * (yC - xy_pts[0, 1])]
        xy_pts[2] = xy_pts[1].copy()

        xy_pts[3] = [xy_pts[5, 0] + curviness * (xC - xy_pts[5, 0]), xy_pts[5, 1] + curviness * (yC - xy_pts[5, 1])]
        xy_pts[4] = xy_pts[3].copy()

        return xy_pts

    @staticmethod
    def calcVal(u, pts):
        p = np.array(pts)
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

        val = (p[0] * (u0 * -1 + t2 - t3 + t4 - t5 + u5 * 1) +
               p[1] * (t9 - t10 + t11 + u3 * -20 + t5) +
               p[2] * (-t15 + u1 * 30 - t11 + t4) +
               p[3] * (t15 - t10 + t3) +
               p[4] * (-t9 + t2) + p[5] * u0 * 1)
        return val
    
    @staticmethod
    def calcDerivU(u, pts, order):
        p0 = pts[0]
        p1 = pts[1]
        p2 = pts[2]
        p3 = pts[3]
        p4 = pts[4]
        p5 = pts[5]

        if order == 1:
            t1 = u * u  # u ^ 2
            t2 = t1 * t1  # t1 ^ 2
            t4 = t1 * u
            t5 = t4 * 0.20e2
            t6 = t1 * 0.30e2
            t7 = u * 0.20e2
            t10 = t2 * 0.25e2
            t11 = t4 * 0.80e2
            t12 = t1 * 0.90e2
            t16 = t2 * 0.50e2
            val = (p0 * (t2 * (-0.5e1) + t5 - t6 + t7 - 0.5e1) +
                p1 * (t10 - t11 + t12 + u * (-0.40e2) + 0.5e1) +
                p2 * (-t16 + t4 * 0.120e3 - t12 + t7) +
                p3 * (t16 - t11 + t6) +
                p4 * (-t10 + t5) +
                p5 * t2 * 0.5e1)
        elif order == 2:
            t1 = u * u  # u ^ 2
            t2 = t1 * u
            t4 = t1 * 0.60e2
            t5 = u * 0.60e2
            t8 = t2 * 0.100e3
            t9 = t1 * 0.240e3
            t10 = u * 0.180e3
            t13 = t2 * 0.200e3
            val = (p0 * (t2 * (-0.20e2) + t4 - t5 + 0.20e2) +
                p1 * (t8 - t9 + t10 - 0.40e2) +
                p2 * (-t13 + t1 * 0.360e3 - t10 + 0.20e2) +
                p3 * (t13 - t9 + t5) +
                p4 * (-t8 + t4) +
                p5 * t2 * 0.20e2)
        elif order == 3:
            t1 = u * u  # u ^ 2
            t3 = u * 0.120e3
            t6 = t1 * 0.300e3
            t7 = u * 0.480e3
            t10 = t1 * 0.600e3
            val = (p0 * (t1 * (-0.60e2) + t3 - 0.60e2) +
                p1 * (t6 - t7 + 0.180e3) +
                p2 * (-t10 + u * 0.720e3 - 0.180e3) +
                p3 * (t10 - t7 + 0.60e2) +
                p4 * (-t6 + t3) +
                p5 * t1 * 0.60e2)
        elif order == 4:
            t4 = u * 0.600e3
            t7 = u * 0.1200e4
            val = (p0 * (u * (-0.120e3) + 0.120e3) +
                p1 * (t4 - 0.480e3) +
                p2 * (-t7 + 0.720e3) +
                p3 * (t7 - 0.480e3) +
                p4 * (-t4 + 0.120e3) +
                p5 * u * 0.120e3)
        elif order == 5:
            val = (p0 * (-0.120e3) +
                p1 * 0.600e3 +
                p2 * (-0.1200e4) +
                p3 * 0.1200e4 +
                p4 * (-0.600e3) +
                p5 * 0.120e3)
        else:
            val = 0

        return val
    
    @staticmethod
    def calcDerivDYDX(u, xpts, ypts, order):
        
        # Compute the derivative d^n y / dx^n
        if order == 1:  # Calculate dy/dx
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            dydx = dydu / dxdu
            val = dydx
        elif order == 2:  # Calculate d^2y/dx^2
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            d2xdu2 = QuinticBezierCurve.calcDerivU(u, xpts, 2)
            d2ydu2 = QuinticBezierCurve.calcDerivU(u, ypts, 2)
            t1 = 1.0 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            d2ydx2 = (d2ydu2 * t1 - dydu / t3 * d2xdu2) * t1
            val = d2ydx2
        elif order == 3:  # Calculate d^3y/dx^3
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            d2xdu2 = QuinticBezierCurve.calcDerivU(u, xpts, 2)
            d2ydu2 = QuinticBezierCurve.calcDerivU(u, ypts, 2)
            d3xdu3 = QuinticBezierCurve.calcDerivU(u, xpts, 3)
            d3ydu3 = QuinticBezierCurve.calcDerivU(u, ypts, 3)
            t1 = 1.0 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            t4 = 1.0 / t3
            t11 = d2xdu2 * d2xdu2  # (d2xdu2 ^ 2)
            t14 = dydu * t4
            d3ydx3 = ((d3ydu3 * t1 - 2 * d2ydu2 * t4 * d2xdu2
                    + 2 * dydu / t3 / dxdu * t11 - t14 * d3xdu3) * t1
                    - (d2ydu2 * t1 - t14 * d2xdu2) * t4 * d2xdu2) * t1
            val = d3ydx3
        elif order == 4:  # Calculate d^4y/dx^4
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            d2xdu2 = QuinticBezierCurve.calcDerivU(u, xpts, 2)
            d2ydu2 = QuinticBezierCurve.calcDerivU(u, ypts, 2)
            d3xdu3 = QuinticBezierCurve.calcDerivU(u, xpts, 3)
            d3ydu3 = QuinticBezierCurve.calcDerivU(u, ypts, 3)
            d4xdu4 = QuinticBezierCurve.calcDerivU(u, xpts, 4)
            d4ydu4 = QuinticBezierCurve.calcDerivU(u, ypts, 4)
            t1 = 1.0 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            t4 = 1.0 / t3
            t9 = 1.0 / t3 / dxdu
            t11 = d2xdu2 * d2xdu2  # (d2xdu2 ^ 2)
            t14 = d2ydu2 * t4
            t17 = t3 * t3  # (t3 ^ 2)
            t23 = dydu * t9
            t27 = dydu * t4
            t37 = d3ydu3 * t1 - 2 * t14 * d2xdu2 + 2 * t23 * t11 - t27 * d3xdu3
            t43 = d2ydu2 * t1 - t27 * d2xdu2
            t47 = t43 * t4
            d4ydx4 = (((d4ydu4 * t1 - 3 * d3ydu3 * t4 * d2xdu2
                        + 6 * d2ydu2 * t9 * t11 - 3 * t14 * d3xdu3
                        - 6 * dydu / t17 * t11 * d2xdu2 + 6 * t23 * d2xdu2 * d3xdu3
                        - t27 * d4xdu4) * t1 - 2 * t37 * t4 * d2xdu2
                        + 2 * t43 * t9 * t11 - t47 * d3xdu3) * t1
                    - (t37 * t1 - t47 * d2xdu2) * t4 * d2xdu2) * t1
            val = d4ydx4
        elif order == 5:  # Calculate d^5y/dx^5
            # Assuming QuinticBezierCurve is a class with the method calcDerivU
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            d2xdu2 = QuinticBezierCurve.calcDerivU(u, xpts, 2)
            d2ydu2 = QuinticBezierCurve.calcDerivU(u, ypts, 2)
            d3xdu3 = QuinticBezierCurve.calcDerivU(u, xpts, 3)
            d3ydu3 = QuinticBezierCurve.calcDerivU(u, ypts, 3)
            d4xdu4 = QuinticBezierCurve.calcDerivU(u, xpts, 4)
            d4ydu4 = QuinticBezierCurve.calcDerivU(u, ypts, 4)
            d5xdu5 = QuinticBezierCurve.calcDerivU(u, xpts, 5)
            d5ydu5 = QuinticBezierCurve.calcDerivU(u, ypts, 5)

            t1 = 1 / dxdu
            t3 = dxdu * dxdu  # dxdu ^ 2
            t4 = 1 / t3
            t9 = 0.1 / (t3 * dxdu)
            t11 = d2xdu2 * d2xdu2  # (d2xdu2 ^ 2)
            t14 = d3ydu3 * t4
            t17 = t3 * t3  # (t3 ^ 2)
            t18 = 1 / t17
            t20 = t11 * d2xdu2
            t23 = d2ydu2 * t9
            t24 = d2xdu2 * d3xdu3
            t27 = d2ydu2 * t4
            t33 = t11 * t11  # (t11 ^ 2)
            t36 = dydu * t18
            t40 = dydu * t9
            t41 = d3xdu3 * d3xdu3  # (d3xdu3 ^ 2)
            t47 = dydu * t4

            t49 = (d5ydu5 * t1 - 4 * d4ydu4 * t4 * d2xdu2 + 12 * d3ydu3 * t9 * t11 
                - 6 * t14 * d3xdu3 - 24 * d2ydu2 * t18 * t20 + 24 * t23 * t24 - 
                4 * t27 * d4xdu4 + 24 * dydu / (t17 * dxdu) * t33 - 36 * t36 * t11 * d3xdu3 
                + 6 * t40 * t41 + 8 * t40 * d2xdu2 * d4xdu4 - t47 * d5xdu5)

            t63 = (d4ydu4 * t1 - 3 * t14 * d2xdu2 + 6 * t23 * t11
                - 3 * t27 * d3xdu3 - 6 * t36 * t20 + 6 * t40 * t24 - t47 * d4xdu4)

            t73 = (d3ydu3 * t1 - 2 * t27 * d2xdu2 + 2 * t40 * t11 - t47 * d3xdu3)
            t77 = t73 * t4
            t82 = d2ydu2 * t1 - t47 * d2xdu2
            t86 = t82 * t9
            t89 = t82 * t4

            t99 = (t63 * t1 - 2 * t77 * d2xdu2 + 2 * t86 * t11 - t89 * d3xdu3)
            t105 = t73 * t1 - t89 * d2xdu2
            t109 = t105 * t4

            d5ydx5 = (((t49 * t1
                        - 3 * t63 * t4 * d2xdu2
                        + 6 * t73 * t9 * t11
                        - 3 * t77 * d3xdu3
                        - 6 * t82 * t18 * t20
                        + 6 * t86 * t24
                        - t89 * d4xdu4) * t1
                    - 2 * t99 * t4 * d2xdu2
                    + 2 * t105 * t9 * t11
                    - t109 * d3xdu3) * t1
                    - (t99 * t1 - t109 * d2xdu2) * t4 * d2xdu2) * t1

            val = d5ydx5

        elif order == 6:  # Calculate d^6y/dx^6
            # Assuming QuinticBezierCurve is a class with the method calcDerivU
            dxdu = QuinticBezierCurve.calcDerivU(u, xpts, 1)
            dydu = QuinticBezierCurve.calcDerivU(u, ypts, 1)
            d2xdu2 = QuinticBezierCurve.calcDerivU(u, xpts, 2)
            d2ydu2 = QuinticBezierCurve.calcDerivU(u, ypts, 2)
            d3xdu3 = QuinticBezierCurve.calcDerivU(u, xpts, 3)
            d3ydu3 = QuinticBezierCurve.calcDerivU(u, ypts, 3)
            d4xdu4 = QuinticBezierCurve.calcDerivU(u, xpts, 4)
            d4ydu4 = QuinticBezierCurve.calcDerivU(u, ypts, 4)
            d5xdu5 = QuinticBezierCurve.calcDerivU(u, xpts, 5)
            d5ydu5 = QuinticBezierCurve.calcDerivU(u, ypts, 5)
            d6xdu6 = QuinticBezierCurve.calcDerivU(u, xpts, 6)
            d6ydu6 = QuinticBezierCurve.calcDerivU(u, ypts, 6)

            # Compute intermediate variables
            t1 = dxdu * dxdu  # dxdu ^ 2
            t3 = 1 / (t1 * dxdu)
            t5 = d2xdu2 * d2xdu2  # (d2xdu2 ^ 2)
            t8 = t1 * t1  # (t1 ^ 2)
            t9 = 1 / t8
            t11 = t5 * d2xdu2
            t14 = d3ydu3 * t3
            t15 = d2xdu2 * d3xdu3
            t19 = 1 / (t8 * dxdu)
            t21 = t5 * t5  # (t5 ^ 2)
            t24 = d2ydu2 * t9
            t25 = t5 * d3xdu3
            t28 = d2ydu2 * t3
            t29 = d3xdu3 * d3xdu3  # (d3xdu3 ^ 2)
            t32 = d2xdu2 * d4xdu4
            t41 = dydu * t19
            t45 = dydu * t9
            t49 = dydu * t3
            t56 = 1 / dxdu
            t61 = 1 / t1
            t62 = dydu * t61
            t67 = d4ydu4 * t61
            t70 = d2ydu2 * t61
            t73 = d3ydu3 * t61

            # Compute the value
            t76 = (20 * d4ydu4 * t3 * t5
                - 60 * d3ydu3 * t9 * t11
                + 60 * t14 * t15
                + 120 * d2ydu2 * t19 * t21
                - 180 * t24 * t25
                + 30 * t28 * t29
                + 40 * t28 * t32
                - 120 * dydu / (t8 * t1) * t21 * d2xdu2
                + 240 * t41 * t11 * d3xdu3
                - 60 * t45 * t5 * d4xdu4
                + 20 * t49 * d3xdu3 * d4xdu4
                + 10 * t49 * d2xdu2 * d5xdu5
                + d6ydu6 * t56
                - 90 * t45 * d2xdu2 * t29
                - t62 * d6ydu6
                - 5 * d5ydu5 * t61 * d2xdu2
                - 10 * t67 * d3xdu3
                - 5 * t70 * d5xdu5
                - 10 * t73 * d4xdu4)

            t100 = (d5ydu5 * t56
                    - 4 * t67 * d2xdu2
                    + 12 * t14 * t5
                    - 6 * t73 * d3xdu3
                    - 24 * t24 * t11
                    + 24 * t28 * t15
                    - 4 * t70 * d4xdu4
                    + 24 * t41 * t21
                    - 36 * t45 * t25
                    + 6 * t49 * t29
                    + 8 * t49 * t32
                    - t62 * d5xdu5)

            t116 = (d4ydu4 * t56
                    - 3 * t73 * d2xdu2
                    + 6 * t28 * t5
                    - 3 * t70 * d3xdu3
                    - 6 * t45 * t11
                    + 6 * t49 * t15
                    - t62 * d4xdu4)

            t120 = t116 * t61
            t129 = (d3ydu3 * t56
                    - 2 * t70 * d2xdu2
                    + 2 * t49 * t5
                    - t62 * d3xdu3)
            t133 = t129 * t3
            t136 = t129 * t61
            t141 = (d2ydu2 * t56
                    - t62 * d2xdu2)
            t145 = t141 * t9
            t148 = t141 * t3
            t153 = t141 * t61

            t155 = (t76 * t56
                    - 4 * t100 * t61 * d2xdu2
                    + 12 * t116 * t3 * t5
                    - 6 * t120 * d3xdu3
                    - 24 * t129 * t9 * t11
                    + 24 * t133 * t15
                    - 4 * t136 * d4xdu4
                    + 24 * t141 * t19 * t21
                    - 36 * t145 * t25
                    + 6 * t148 * t29
                    + 8 * t148 * t32
                    - t153 * d5xdu5)

            t169 = (t100 * t56
                    - 3 * t120 * d2xdu2
                    + 6 * t133 * t5
                    - 3 * t136 * d3xdu3
                    - 6 * t145 * t11
                    + 6 * t148 * t15
                    - t153 * d4xdu4)

            t179 = (t116 * t56
                    - 2 * t136 * d2xdu2
                    + 2 * t148 * t5
                    - t153 * d3xdu3)
            t183 = t179 * t61
            t188 = t129 * t56 - t153 * d2xdu2
            t192 = t188 * t3
            t195 = t188 * t61

            t205 = (t169 * t56
                    - 2 * t183 * d2xdu2
                    + 2 * t192 * t5
                    - t195 * d3xdu3)

            t211 = t179 * t56 - t195 * d2xdu2
            t215 = t211 * t61

            d6ydx6 = (((t155 * t56
                        - 3 * t169 * t61 * d2xdu2
                        + 6 * t179 * t3 * t5
                        - 3 * t183 * d3xdu3
                        - 6 * t188 * t9 * t11
                        + 6 * t192 * t15
                        - t195 * d4xdu4) * t56
                    - 2 * t205 * t61 * d2xdu2
                    + 2 * t211 * t3 * t5
                    - t215 * d3xdu3) * t56
                    - (t205 * t56 - t215 * d2xdu2) * t61 * d2xdu2) * t56

            val = d6ydx6
        else:
            raise ValueError('Order must be between 1 and 6.')
        
        return val

    @staticmethod
    def clampU(u):
        u = np.max(0.0,min(1.0,u))
        return u
    
    @staticmethod
    def calcU(ax, bezier_pts_x):
        """
        Computes u given x for a Bezier curve.
        
        Parameters:
        ax (float): The x value for which u is computed.
        bezier_pts_x (numpy.ndarray): 2D array of Bezier control points for x-coordinates.
        
        Returns:
        float: The computed u value.
        """
        # Check to make sure that ax is in the curve domain
        min_x = np.min(bezier_pts_x)
        max_x = np.max(bezier_pts_x)
        if not (min_x <= ax <= max_x):
            raise ValueError('Input ax is not in the domain of the Bezier curve specified by the control points in bezier_pts_x.')
        
        # We want to find u such that x(u) - ax = 0
        def x_ax(u):
            x = QuinticBezierCurve.calcVal(u, bezier_pts_x)
            f = x - ax
            J = QuinticBezierCurve.calcDerivU(u,bezier_pts_x,1)
            return [f,J]
        
        # Initial guess
        u_init = (ax - min_x) / (max_x - min_x)
        
        # Use Newton's method to find the root
        temp = fzero_newton(x_ax, u_init, tol=1e-9, kmax=10)
        return temp[0]

    @staticmethod
    def calcIndex(x, bezierPtsX):
        idx = 0
        flag_found = False

        # Iterate over columns in bezier_pts_x
        for i in range(bezierPtsX.shape[1]):
            if bezierPtsX[0, i] <= x < bezierPtsX[5, i]:
                idx = i
                flag_found = True
                break

        # Check if x is identically the last point
        if not flag_found and x == bezierPtsX[5, -1]:
            idx = bezierPtsX.shape[1] - 1
            flag_found = True

        # Optional: Raise an assertion error if the value x is not within the Bezier curve set
        # assert flag_found, 'A value of x was used that is not within the Bezier curve set.'
        
        return idx