import numpy as np
from QuinticBezierCurve import *

class SmoothSegmentedFunction:
    def __init__(self, mX, mY, x0, x1, y0, y1, dydx0, dydx1, computeIntegral, intx0x1):
        self.mXVec_ = mX
        self.mYVec_ = mY
        self.x0_ = x0
        self.x1_ = x1
        self.y0_ = y0
        self.y1_ = y1
        self.dydx0_ = dydx0
        self.dydx1_ = dydx1
        self.computeIntegral_ = computeIntegral
        self.intx0x1_ = intx0x1

        assert not computeIntegral

        self.numBezierSections_ = mX.shape[1]

    def getDomains(self):
        return np.hstack((self.mXVec_[0, :], self.mXVec_[-1, -1]))

    def calcValue(self, x):
        if (isinstance(x,(float,int))):
            x = [x]
        yVal = np.zeros_like(x)
        for k, xk in enumerate(x):
            if self.x0_ <= xk <= self.x1_:
                idx = QuinticBezierCurve.calcIndex(xk, self.mXVec_)
                u = QuinticBezierCurve.calcU(xk, self.mXVec_[:, idx])
                yVal[k] = QuinticBezierCurve.calcVal(u, self.mYVec_[:, idx])
            else:
                if xk < self.x0_:
                    yVal[k] = self.y0_ + self.dydx0_ * (xk - self.x0_)
                else:
                    yVal[k] = self.y1_ + self.dydx1_ * (xk - self.x1_)
        if (len(yVal) == 1):
            yVal = yVal[0]
        return yVal

    def calcDerivative(self, x, order):
        if order == 0:
            return self.calcValue(x)
        else:
            yVal = np.zeros_like(x)
            for k, xk in enumerate(x):
                if self.x0_ <= xk <= self.x1_:
                    idx = QuinticBezierCurve.calcIndex(xk, self.mXVec_)
                    mXVec_idx = self.mXVec_[:, idx]
                    mYVec_idx = self.mYVec_[:, idx]
                    u = QuinticBezierCurve.calcU(xk, mXVec_idx)
                    yVal[k] = QuinticBezierCurve.calcDerivDYDX(u, mXVec_idx, mYVec_idx, order)
                else:
                    if order == 1:
                        yVal[k] = self.dydx0_ if xk < self.x0_ else self.dydx1_
                    else:
                        yVal[k] = 0
            return yVal

    def calcValDeriv(self, x):
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        for k in range(0,len(x)):
            xk = x[k]
            if self.x0_ <= xk <= self.x1_:
                idx = QuinticBezierCurve.calcIndex(xk, self.mXVec_)
                mXVec_idx = self.mXVec_[:, idx]
                mYVec_idx = self.mYVec_[:, idx]
                u = QuinticBezierCurve.calcU(xk, mXVec_idx)
                y0[k] = QuinticBezierCurve.calcVal(u, mYVec_idx)
                y1[k] = QuinticBezierCurve.calcDerivDYDX(u, mXVec_idx, mYVec_idx, 1)
                y2[k] = QuinticBezierCurve.calcDerivDYDX(u, mXVec_idx, mYVec_idx, 2)
            else:
                if xk < self.x0_:
                    y0[k] = self.y0_ + self.dydx0_ * (xk - self.x0_)
                    y1[k] = self.dydx0_
                else:
                    y0[k] = self.y1_ + self.dydx1_ * (xk - self.x1_)
                    y1[k] = self.dydx1_
                y2[k] = 0
        return [y0, y1, y2]

    @staticmethod
    def scaleCurviness(curviness):
        return 0.1 + 0.8 * curviness

    @staticmethod
    def createFiberForceLengthCurve(eZero, eIso, kLow, kIso, curviness, computeIntegral):
        assert eIso > eZero, 'eIso must be greater than eZero'
        assert kIso > 1.0 / (eIso - eZero), 'kIso must be greater than 1/(eIso-eZero)'
        assert 0 < kLow < 1 / (eIso - eZero), 'kLow must be between 0 and 1/(eIso-eZero)'
        assert 0 <= curviness <= 1, 'curviness must be between 0 and 1'

        c = SmoothSegmentedFunction.scaleCurviness(curviness)
        xZero = 1 + eZero
        yZero = 0

        xIso = 1 + eIso
        yIso = 1

        deltaX = min(0.1 * (1.0 / kIso), 0.1 * (xIso - xZero))

        xLow = xZero + deltaX
        xfoot = xZero + 0.5 * (xLow - xZero)
        yfoot = 0
        yLow = yfoot + kLow * (xLow - xfoot)

        p0 = QuinticBezierCurve.calcCornerControlPoints(xZero, yZero, 0, xLow, yLow, kLow, c)
        p1 = QuinticBezierCurve.calcCornerControlPoints(xLow, yLow, kLow, xIso, yIso, kIso, c)

        mX = np.zeros((6, 2))
        mY = np.zeros((6, 2))

        mX[:, 0] = p0[:, 0]
        mY[:, 0] = p0[:, 1]

        mX[:, 1] = p1[:, 0]
        mY[:, 1] = p1[:, 1]

        return SmoothSegmentedFunction(mX, mY, xZero, xIso, yZero, yIso, 0.0, kIso, computeIntegral, True)

    def createTendonForceLengthCurve(eIso, kIso, fToe, curviness, compute_integral):
        # Check the input arguments
        assert eIso > 0, f'eIso must be greater than 0, but {eIso} was entered'
        assert 0 < fToe < 1, f'fToe must be greater than 0 and less than 1, but {fToe} was entered'
        assert kIso > (1 / eIso), f'kIso must be greater than 1/eIso, ({1 / eIso}), but kIso ({kIso}) was entered'
        assert 0 <= curviness <= 1, f'curviness must be between 0.0 and 1.0, but {curviness} was entered'

        # Translate the user parameters to quintic Bezier points
        c = SmoothSegmentedFunction.scaleCurviness(curviness)
        x0, y0, dydx0 = 1.0, 0, 0
        xIso, yIso, dydxIso = 1.0 + eIso, 1, kIso
        
        # Location where the curved section becomes linear
        yToe = fToe
        xToe = (yToe - 1) / kIso + xIso
        
        # To limit the 2nd derivative of the toe region the line it tends to
        # has to intersect the x axis to the right of the origin
        xFoot = 1.0 + (xToe - 1.0) / 10.0
        yFoot = 0
        
        # Compute the location of the corner formed by the average slope of the
        # toe and the slope of the linear section
        yToeMid = yToe * 0.5
        xToeMid = (yToeMid - yIso) / kIso + xIso
        dydxToeMid = (yToeMid - yFoot) / (xToeMid - xFoot)
        
        # Compute the location of the control point to the left of the corner
        xToeCtrl = xFoot + 0.5 * (xToeMid - xFoot)
        yToeCtrl = yFoot + dydxToeMid * (xToeCtrl - xFoot)
        
        # Compute the Quintic Bezier control points
        p0 = QuinticBezierCurve.calcCornerControlPoints(x0, y0, dydx0, xToeCtrl, yToeCtrl, dydxToeMid, c)
        p1 = QuinticBezierCurve.calcCornerControlPoints(xToeCtrl, yToeCtrl, dydxToeMid, xToe, yToe, dydxIso, c)

        mX = np.zeros((6, 2))
        mY = np.zeros((6, 2))
        
        mX[:, 0] = p0[:, 0]
        mY[:, 0] = p0[:, 1]
        mX[:, 1] = p1[:, 0]
        mY[:, 1] = p1[:, 1]

        # Instantiate a muscle curve object
        return SmoothSegmentedFunction(
            mX, mY,
            x0, xToe,
            y0, yToe,
            dydx0, dydxIso,
            compute_integral,
            True
        )

    def createFiberForceVelocityCurve(fmaxE, dydxC, dydxNearC, dydxIso, dydxE, dydxNearE, conc_curviness, ecc_curviness, compute_integral):
        # Ensure that the inputs are within a valid range
        assert fmaxE > 1.0, 'fmaxE must be greater than 1'
        assert 0.0 <= dydxC < 1, 'dydxC must be greater than or equal to 0 and less than 1'
        assert dydxNearC > dydxC and dydxNearC <= 1, 'dydxNearC must be greater than or equal to 0 and less than 1'
        assert dydxIso > 1, f'dydxIso must be greater than (fmaxE-1)/1 ({(fmaxE - 1.0) / 1.0})'
        assert 0.0 <= dydxE < (fmaxE - 1), f'dydxE must be greater than or equal to 0 and less than fmaxE-1 ({fmaxE - 1})'
        assert dydxNearE >= dydxE and dydxNearE < (fmaxE - 1), f'dydxNearE must be greater than or equal to dydxE and less than fmaxE-1 ({fmaxE - 1})'
        assert 0 <= conc_curviness <= 1, 'concCurviness must be between 0 and 1'
        assert 0 <= ecc_curviness <= 1, 'eccCurviness must be between 0 and 1'
        
        # Translate the users parameters into Bezier point locations
        cC = SmoothSegmentedFunction.scaleCurviness(conc_curviness)
        cE = SmoothSegmentedFunction.scaleCurviness(ecc_curviness)
        
        # Compute the concentric control point locations
        xC, yC = -1, 0
        xNearC = -0.9
        yNearC = yC + 0.5 * dydxNearC * (xNearC - xC) + 0.5 * dydxC * (xNearC - xC)
        
        xIso, yIso = 0, 1
        
        xE, yE = 1, fmaxE
        xNearE = 0.9
        yNearE = yE + 0.5 * dydxNearE * (xNearE - xE) + 0.5 * dydxE * (xNearE - xE)
        
        concPts1 = QuinticBezierCurve.calcCornerControlPoints(xC, yC, dydxC, xNearC, yNearC, dydxNearC, cC)
        concPts2 = QuinticBezierCurve.calcCornerControlPoints(xNearC, yNearC, dydxNearC, xIso, yIso, dydxIso, cC)
        eccPts1 = QuinticBezierCurve.calcCornerControlPoints(xIso, yIso, dydxIso, xNearE, yNearE, dydxNearE, cE)
        eccPts2 = QuinticBezierCurve.calcCornerControlPoints(xNearE, yNearE, dydxNearE, xE, yE, dydxE, cE)
        
        mX = np.zeros((6, 4))
        mY = np.zeros((6, 4))
        
        mX[:, 0] = concPts1[:, 0]
        mX[:, 1] = concPts2[:, 0]
        mX[:, 2] = eccPts1[:, 0]
        mX[:, 3] = eccPts2[:, 0]
        
        mY[:, 0] = concPts1[:, 1]
        mY[:, 1] = concPts2[:, 1]
        mY[:, 2] = eccPts1[:, 1]
        mY[:, 3] = eccPts2[:, 1]

        return SmoothSegmentedFunction(
            mX, mY,
            xC, xE,
            yC, yE,
            dydxC, dydxE,
            compute_integral,
            True
        )

    def createFiberForceVelocityInverseCurve(fmaxE, dydxC, dydxNearC, dydxIso, dydxE, dydxNearE, conc_curviness, ecc_curviness, compute_integral):
        # Ensure that the inputs are within a valid range
        root_eps = np.sqrt(np.finfo(float).eps)
        assert fmaxE > 1.0, 'fmaxE must be greater than 1'
        assert root_eps < dydxC < 1, 'dydxC must be greater than 0 and less than 1'
        assert dydxNearC > dydxC and dydxNearC < 1, 'dydxNearC must be greater than 0 and less than 1'
        assert dydxIso > 1, 'dydxIso must be greater than or equal to 1'
        assert root_eps < dydxE < (fmaxE - 1), f'dydxE must be greater than or equal to 0 and less than fmaxE-1 ({fmaxE - 1})'
        assert dydxNearE >= dydxE and dydxNearE < (fmaxE - 1), f'dydxNearE must be greater than or equal to dydxE and less than fmaxE-1 ({fmaxE - 1})'
        assert 0 <= conc_curviness <= 1, 'concCurviness must be between 0 and 1'
        assert 0 <= ecc_curviness <= 1, 'eccCurviness must be between 0 and 1'
        
        # Translate the users parameters into Bezier point locations
        cC = SmoothSegmentedFunction.scaleCurviness(conc_curviness)
        cE = SmoothSegmentedFunction.scaleCurviness(ecc_curviness)

        # Compute the concentric control point locations
        xC, yC = -1, 0
        xNearC = -0.9
        yNearC = yC + 0.5 * dydxNearC * (xNearC - xC) + 0.5 * dydxC * (xNearC - xC)
        
        xIso, yIso = 0, 1
        
        xE, yE = 1, fmaxE
        xNearE = 0.9
        yNearE = yE + 0.5 * dydxNearE * (xNearE - xE) + 0.5 * dydxE * (xNearE - xE)
        
        concPts1 = QuinticBezierCurve.calcCornerControlPoints(xC, yC, dydxC, xNearC, yNearC, dydxNearC, cC)
        concPts2 = QuinticBezierCurve.calcCornerControlPoints(xNearC, yNearC, dydxNearC, xIso, yIso, dydxIso, cC)
        eccPts1 = QuinticBezierCurve.calcCornerControlPoints(xIso, yIso, dydxIso, xNearE, yNearE, dydxNearE, cE)
        eccPts2 = QuinticBezierCurve.calcCornerControlPoints(xNearE, yNearE, dydxNearE, xE, yE, dydxE, cE)
        
        mX = np.zeros((6, 4))
        mY = np.zeros((6, 4))
        
        mX[:, 0] = concPts1[:, 0]
        mX[:, 1] = concPts2[:, 0]
        mX[:, 2] = eccPts1[:, 0]
        mX[:, 3] = eccPts2[:, 0]
        
        mY[:, 0] = concPts1[:, 1]
        mY[:, 1] = concPts2[:, 1]
        mY[:, 2] = eccPts1[:, 1]
        mY[:, 3] = eccPts2[:, 1]

        return SmoothSegmentedFunction(
            mY, mX,
            yC, yE,
            xC, xE,
            1 / dydxC, 1 / dydxE,
            compute_integral,
            True
        )

    def createFiberActiveForceLengthCurve(x0, x1, x2, x3, ylow, dydx, curviness, compute_integral):
        # Ensure that the inputs are within a valid range
        root_eps = np.sqrt(np.finfo(float).eps)
        assert x0 >= 0 and x1 > x0 + root_eps and x2 > x1 + root_eps and x3 > x2 + root_eps, 'This must be true: 0 < lce0 < lce1 < lce2 < lce3'
        assert ylow >= 0, 'shoulderVal must be greater than, or equal to 0'
        dydx_upper_bound = (1 - ylow) / (x2 - x1)
        assert 0 <= dydx < dydx_upper_bound, f'plateauSlope must be greater than 0 and less than {dydx_upper_bound}'
        assert 0 <= curviness <= 1, 'curviness must be between 0 and 1'
        
        # Translate the users parameters into Bezier curves
        c = SmoothSegmentedFunction.scaleCurviness(curviness)

        # The active force length curve is made up of 5 elbow shaped sections.
        # Compute the locations of the joining point of each elbow section.
        
        # Calculate the location of the shoulder
        xDelta = 0.05 * x2  # half the width of the sarcomere
        
        xs = (x2 - xDelta)  # x1 + 0.75*(x2-x1)
        
        # Calculate the intermediate points located on the ascending limb
        y0, dydx0 = 0, 0
        y1 = 1 - dydx * (xs - x1)
        dydx01 = 1.25 * (y1 - y0) / (x1 - x0)
        
        x01 = x0 + 0.5 * (x1 - x0)
        y01 = y0 + 0.5 * (y1 - y0)
        
        # Calculate the intermediate points of the shallow ascending plateau
        x1s = x1 + 0.5 * (xs - x1)
        y1s = y1 + 0.5 * (1 - y1)
        dydx1s = dydx
        
        # x2 entered
        y2, dydx2 = 1, 0
        
        # Descending limb
        # x3 entered
        y3, dydx3 = 0, 0
        
        x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta))
        y23 = y2 + 0.5 * (y3 - y2)
        dydx23 = (y3 - y2) / ((x3 - xDelta) - (x2 + xDelta))
        
        # Compute the locations of the control points
        p0 = QuinticBezierCurve.calcCornerControlPoints(x0, ylow, dydx0, x01, y01, dydx01, c)
        p1 = QuinticBezierCurve.calcCornerControlPoints(x01, y01, dydx01, x1s, y1s, dydx1s, c)
        p2 = QuinticBezierCurve.calcCornerControlPoints(x1s, y1s, dydx1s, x2, y2, dydx2, c)
        p3 = QuinticBezierCurve.calcCornerControlPoints(x2, y2, dydx2, x23, y23, dydx23, c)
        p4 = QuinticBezierCurve.calcCornerControlPoints(x23, y23, dydx23, x3, ylow, dydx3, c)
        
        mX = np.zeros((6, 5))
        mY = np.zeros((6, 5))
        
        mX[:, 0] = p0[:, 0]
        mX[:, 1] = p1[:, 0]
        mX[:, 2] = p2[:, 0]
        mX[:, 3] = p3[:, 0]
        mX[:, 4] = p4[:, 0]
        
        mY[:, 0] = p0[:, 1]
        mY[:, 1] = p1[:, 1]
        mY[:, 2] = p2[:, 1]
        mY[:, 3] = p3[:, 1]
        mY[:, 4] = p4[:, 1]
        
        return SmoothSegmentedFunction(
            mX, mY,
            x0, x3,
            ylow, ylow,
            0, 0,
            compute_integral,
            True
        )
