#include "SmoothSegmentedFunction.h"

SmoothSegmentedFunction::SmoothSegmentedFunction(
    const std::vector<std::vector<double>>& mX,
    const std::vector<std::vector<double>>& mY,
    double x0, double x1, double y0, double y1,
    double dydx0, double dydx1,
    bool computeIntegral, bool intx0x1)
    : mXVec_(mX), mYVec_(mY), x0_(x0), x1_(x1), y0_(y0), y1_(y1),
      dydx0_(dydx0), dydx1_(dydx1), computeIntegral_(computeIntegral), intx0x1_(intx0x1)
{
    assert(!computeIntegral && "Integration not yet supported");
    numBezierSections_ = mX.size();
}

std::vector<double> SmoothSegmentedFunction::getDomains() const
{
    std::vector<double> domains;
    for (int i = 0; i < numBezierSections_; ++i) {
        domains.push_back(mXVec_[i][0]);
    }
    domains.push_back(mXVec_[numBezierSections_ - 1][5]);
    return domains;
}

double SmoothSegmentedFunction::calcValue(double x) const
{
    double yVal;
    
    if (x0_ <= x && x <= x1_) {
        // Find the Bezier section that contains x
        int idx = -1;
        for (int i = 0; i < numBezierSections_; ++i) {
            if (mXVec_[i][0] <= x && x <= mXVec_[i][5]) {
                idx = i;
                break;
            }
        }
        
        if (idx == -1) {
            // Handle edge case: x is exactly at x1_
            if (std::abs(x - x1_) < std::numeric_limits<double>::epsilon()) {
                idx = numBezierSections_ - 1;
            } else {
                throw std::runtime_error("Value not within Bezier curve range");
            }
        }
        
        // Extract x control points for this section
        std::vector<double> xPts(6);
        for (int i = 0; i < 6; ++i) {
            xPts[i] = mXVec_[idx][i];
        }
        
        // Compute parameter u
        double u = QuinticBezierCurve::calcU(x, xPts);
        
        // Create points array for calcVal
        std::vector<std::array<double, 2>> pts(6);
        for (int i = 0; i < 6; ++i) {
            pts[i] = {0.0, mYVec_[idx][i]};
        }
        
        // Calculate the value
        yVal = QuinticBezierCurve::calcVal(u, pts);
    } else {
        // Outside the curve range, use linear extrapolation
        if (x < x0_) {
            yVal = y0_ + dydx0_ * (x - x0_);
        } else {
            yVal = y1_ + dydx1_ * (x - x1_);
        }
    }
    
    return yVal;
}

double SmoothSegmentedFunction::calcDerivative(double x, int order) const
{
    if (order == 0) {
        return calcValue(x);
    }
    
    double yVal;
    
    if (x0_ <= x && x <= x1_) {
        // Find the Bezier section that contains x
        int idx = -1;
        for (int i = 0; i < numBezierSections_; ++i) {
            if (mXVec_[i][0] <= x && x <= mXVec_[i][5]) {
                idx = i;
                break;
            }
        }
        
        if (idx == -1) {
            // Handle edge case: x is exactly at x1_
            if (std::abs(x - x1_) < std::numeric_limits<double>::epsilon()) {
                idx = numBezierSections_ - 1;
            } else {
                throw std::runtime_error("Value not within Bezier curve range");
            }
        }
        
        // Extract x and y control points for this section
        std::vector<double> xPts(6);
        std::vector<double> yPts(6);
        for (int i = 0; i < 6; ++i) {
            xPts[i] = mXVec_[idx][i];
            yPts[i] = mYVec_[idx][i];
        }
        
        // Compute parameter u
        double u = QuinticBezierCurve::calcU(x, xPts);
        
        // Calculate the derivative
        yVal = QuinticBezierCurve::calcDerivDYDX(u, xPts, yPts, order);
    } else {
        // Outside the curve range
        if (order == 1) {
            yVal = (x < x0_) ? dydx0_ : dydx1_;
        } else {
            yVal = 0.0;
        }
    }
    
    return yVal;
}

std::vector<double> SmoothSegmentedFunction::calcValDeriv(double x) const
{
    std::vector<double> result(3, 0.0); // y0, y1, y2
    
    if (x0_ <= x && x <= x1_) {
        // Find the Bezier section that contains x
        int idx = -1;
        for (int i = 0; i < numBezierSections_; ++i) {
            if (mXVec_[i][0] <= x && x <= mXVec_[i][5]) {
                idx = i;
                break;
            }
        }
        
        if (idx == -1) {
            // Handle edge case: x is exactly at x1_
            if (std::abs(x - x1_) < std::numeric_limits<double>::epsilon()) {
                idx = numBezierSections_ - 1;
            } else {
                throw std::runtime_error("Value not within Bezier curve range");
            }
        }
        
        // Extract x and y control points for this section
        std::vector<double> xPts(6);
        std::vector<double> yPts(6);
        for (int i = 0; i < 6; ++i) {
            xPts[i] = mXVec_[idx][i];
            yPts[i] = mYVec_[idx][i];
        }
        
        // Compute parameter u
        double u = QuinticBezierCurve::calcU(x, xPts);
        
        // Create points array for calcVal
        std::vector<std::array<double, 2>> pts(6);
        for (int i = 0; i < 6; ++i) {
            pts[i] = {0.0, yPts[i]};
        }
        
        // Calculate the value and derivatives
        result[0] = QuinticBezierCurve::calcVal(u, pts);
        result[1] = QuinticBezierCurve::calcDerivDYDX(u, xPts, yPts, 1);
        result[2] = QuinticBezierCurve::calcDerivDYDX(u, xPts, yPts, 2);
    } else {
        // Outside the curve range, use linear extrapolation
        if (x < x0_) {
            result[0] = y0_ + dydx0_ * (x - x0_);
            result[1] = dydx0_;
        } else {
            result[0] = y1_ + dydx1_ * (x - x1_);
            result[1] = dydx1_;
        }
        result[2] = 0.0;
    }
    
    return result;
}

double SmoothSegmentedFunction::scaleCurviness(double curviness)
{
    return 0.1 + 0.8 * curviness;
}

SmoothSegmentedFunction SmoothSegmentedFunction::createFiberForceLengthCurve(
    double eZero, double eIso, double kLow, double kIso, 
    double curviness, bool computeIntegral)
{
    assert(eIso > eZero && "eIso must be greater than eZero");
    assert(kIso > 1.0 / (eIso - eZero) && "kIso must be greater than 1/(eIso-eZero)");
    assert(0 < kLow && kLow < 1 / (eIso - eZero) && "kLow must be between 0 and 1/(eIso-eZero)");
    assert(0 <= curviness && curviness <= 1 && "curviness must be between 0 and 1");
    
    double c = scaleCurviness(curviness);
    double xZero = 1 + eZero;
    double yZero = 0;
    
    double xIso = 1 + eIso;
    double yIso = 1;
    
    double deltaX = std::min(0.1 * (1.0 / kIso), 0.1 * (xIso - xZero));
    
    double xLow = xZero + deltaX;
    double xfoot = xZero + 0.5 * (xLow - xZero);
    double yfoot = 0;
    double yLow = yfoot + kLow * (xLow - xfoot);
    
    // Calculate control points
    auto p0 = QuinticBezierCurve::calcCornerControlPoints(xZero, yZero, 0, xLow, yLow, kLow, c);
    auto p1 = QuinticBezierCurve::calcCornerControlPoints(xLow, yLow, kLow, xIso, yIso, kIso, c);
    
    // Create mX and mY arrays
    std::vector<std::vector<double>> mX(2, std::vector<double>(6));
    std::vector<std::vector<double>> mY(2, std::vector<double>(6));
    
    for (int i = 0; i < 6; ++i) {
        mX[0][i] = p0[i][0];
        mY[0][i] = p0[i][1];
        
        mX[1][i] = p1[i][0];
        mY[1][i] = p1[i][1];
    }
    
    return SmoothSegmentedFunction(mX, mY, xZero, xIso, yZero, yIso, 0.0, kIso, computeIntegral, true);
}

SmoothSegmentedFunction SmoothSegmentedFunction::createTendonForceLengthCurve(
    double eIso, double kIso, double fToe, double curviness, 
    bool computeIntegral)
{
    // Check the input arguments
    assert(eIso > 0 && "eIso must be greater than 0");
    assert(0 < fToe && fToe < 1 && "fToe must be greater than 0 and less than 1");
    assert(kIso > (1 / eIso) && "kIso must be greater than 1/eIso");
    assert(0 <= curviness && curviness <= 1 && "curviness must be between 0.0 and 1.0");
    
    // Translate the user parameters to quintic Bezier points
    double c = scaleCurviness(curviness);
    double x0 = 1.0, y0 = 0, dydx0 = 0;
    double xIso = 1.0 + eIso, yIso = 1, dydxIso = kIso;
    
    // Location where the curved section becomes linear
    double yToe = fToe;
    double xToe = (yToe - 1) / kIso + xIso;
    
    // To limit the 2nd derivative of the toe region the line it tends to
    // has to intersect the x axis to the right of the origin
    double xFoot = 1.0 + (xToe - 1.0) / 10.0;
    double yFoot = 0;
    
    // Compute the location of the corner formed by the average slope of the
    // toe and the slope of the linear section
    double yToeMid = yToe * 0.5;
    double xToeMid = (yToeMid - yIso) / kIso + xIso;
    double dydxToeMid = (yToeMid - yFoot) / (xToeMid - xFoot);
    
    // Compute the location of the control point to the left of the corner
    double xToeCtrl = xFoot + 0.5 * (xToeMid - xFoot);
    double yToeCtrl = yFoot + dydxToeMid * (xToeCtrl - xFoot);
    
    // Compute the Quintic Bezier control points
    auto p0 = QuinticBezierCurve::calcCornerControlPoints(x0, y0, dydx0, xToeCtrl, yToeCtrl, dydxToeMid, c);
    auto p1 = QuinticBezierCurve::calcCornerControlPoints(xToeCtrl, yToeCtrl, dydxToeMid, xToe, yToe, dydxIso, c);
    
    std::vector<std::vector<double>> mX(2, std::vector<double>(6));
    std::vector<std::vector<double>> mY(2, std::vector<double>(6));
    
    for (int i = 0; i < 6; ++i) {
        mX[0][i] = p0[i][0];
        mY[0][i] = p0[i][1];
        
        mX[1][i] = p1[i][0];
        mY[1][i] = p1[i][1];
    }
    
    // Instantiate a muscle curve object
    return SmoothSegmentedFunction(
        mX, mY,
        x0, xToe,
        y0, yToe,
        dydx0, dydxIso,
        computeIntegral,
        true
    );
}

SmoothSegmentedFunction SmoothSegmentedFunction::createFiberForceVelocityCurve(
    double fmaxE, double dydxC, double dydxNearC, double dydxIso,
    double dydxE, double dydxNearE, double concCurviness, 
    double eccCurviness, bool computeIntegral)
{
    // Ensure that the inputs are within a valid range
    assert(fmaxE > 1.0 && "fmaxE must be greater than 1");
    assert(0.0 <= dydxC && dydxC < 1 && "dydxC must be greater than or equal to 0 and less than 1");
    assert(dydxNearC > dydxC && dydxNearC <= 1 && "dydxNearC must be greater than or equal to 0 and less than 1");
    assert(dydxIso > 1 && "dydxIso must be greater than (fmaxE-1)/1");
    assert(0.0 <= dydxE && dydxE < (fmaxE - 1) && "dydxE must be greater than or equal to 0 and less than fmaxE-1");
    assert(dydxNearE >= dydxE && dydxNearE < (fmaxE - 1) && "dydxNearE must be greater than or equal to dydxE and less than fmaxE-1");
    assert(0 <= concCurviness && concCurviness <= 1 && "concCurviness must be between 0 and 1");
    assert(0 <= eccCurviness && eccCurviness <= 1 && "eccCurviness must be between 0 and 1");
    
    // Translate the users parameters into Bezier point locations
    double cC = scaleCurviness(concCurviness);
    double cE = scaleCurviness(eccCurviness);
    
    // Compute the concentric control point locations
    double xC = -1, yC = 0;
    double xNearC = -0.9;
    double yNearC = yC + 0.5 * dydxNearC * (xNearC - xC) + 0.5 * dydxC * (xNearC - xC);
    
    double xIso = 0, yIso = 1;
    
    double xE = 1, yE = fmaxE;
    double xNearE = 0.9;
    double yNearE = yE + 0.5 * dydxNearE * (xNearE - xE) + 0.5 * dydxE * (xNearE - xE);
    
    auto concPts1 = QuinticBezierCurve::calcCornerControlPoints(xC, yC, dydxC, xNearC, yNearC, dydxNearC, cC);
    auto concPts2 = QuinticBezierCurve::calcCornerControlPoints(xNearC, yNearC, dydxNearC, xIso, yIso, dydxIso, cC);
    auto eccPts1 = QuinticBezierCurve::calcCornerControlPoints(xIso, yIso, dydxIso, xNearE, yNearE, dydxNearE, cE);
    auto eccPts2 = QuinticBezierCurve::calcCornerControlPoints(xNearE, yNearE, dydxNearE, xE, yE, dydxE, cE);
    
    std::vector<std::vector<double>> mX(4, std::vector<double>(6));
    std::vector<std::vector<double>> mY(4, std::vector<double>(6));
    
    for (int i = 0; i < 6; ++i) {
        mX[0][i] = concPts1[i][0];
        mX[1][i] = concPts2[i][0];
        mX[2][i] = eccPts1[i][0];
        mX[3][i] = eccPts2[i][0];
        
        mY[0][i] = concPts1[i][1];
        mY[1][i] = concPts2[i][1];
        mY[2][i] = eccPts1[i][1];
        mY[3][i] = eccPts2[i][1];
    }
    
    return SmoothSegmentedFunction(
        mX, mY,
        xC, xE,
        yC, yE,
        dydxC, dydxE,
        computeIntegral,
        true
    );
}

SmoothSegmentedFunction SmoothSegmentedFunction::createFiberForceVelocityInverseCurve(
    double fmaxE, double dydxC, double dydxNearC, double dydxIso,
    double dydxE, double dydxNearE, double concCurviness, 
    double eccCurviness, bool computeIntegral)
{
    // Ensure that the inputs are within a valid range
    double root_eps = std::sqrt(std::numeric_limits<double>::epsilon());
    assert(fmaxE > 1.0 && "fmaxE must be greater than 1");
    assert(root_eps < dydxC && dydxC < 1 && "dydxC must be greater than 0 and less than 1");
    assert(dydxNearC > dydxC && dydxNearC < 1 && "dydxNearC must be greater than 0 and less than 1");
    assert(dydxIso > 1 && "dydxIso must be greater than or equal to 1");
    assert(root_eps < dydxE && dydxE < (fmaxE - 1) && "dydxE must be greater than or equal to 0 and less than fmaxE-1");
    assert(dydxNearE >= dydxE && dydxNearE < (fmaxE - 1) && "dydxNearE must be greater than or equal to dydxE and less than fmaxE-1");
    assert(0 <= concCurviness && concCurviness <= 1 && "concCurviness must be between 0 and 1");
    assert(0 <= eccCurviness && eccCurviness <= 1 && "eccCurviness must be between 0 and 1");
    
    // Translate the users parameters into Bezier point locations
    double cC = scaleCurviness(concCurviness);
    double cE = scaleCurviness(eccCurviness);
    
    // Compute the concentric control point locations
    double xC = -1, yC = 0;
    double xNearC = -0.9;
    double yNearC = yC + 0.5 * dydxNearC * (xNearC - xC) + 0.5 * dydxC * (xNearC - xC);
    
    double xIso = 0, yIso = 1;
    
    double xE = 1, yE = fmaxE;
    double xNearE = 0.9;
    double yNearE = yE + 0.5 * dydxNearE * (xNearE - xE) + 0.5 * dydxE * (xNearE - xE);
    
    auto concPts1 = QuinticBezierCurve::calcCornerControlPoints(xC, yC, dydxC, xNearC, yNearC, dydxNearC, cC);
    auto concPts2 = QuinticBezierCurve::calcCornerControlPoints(xNearC, yNearC, dydxNearC, xIso, yIso, dydxIso, cC);
    auto eccPts1 = QuinticBezierCurve::calcCornerControlPoints(xIso, yIso, dydxIso, xNearE, yNearE, dydxNearE, cE);
    auto eccPts2 = QuinticBezierCurve::calcCornerControlPoints(xNearE, yNearE, dydxNearE, xE, yE, dydxE, cE);
    
    std::vector<std::vector<double>> mX(4, std::vector<double>(6));
    std::vector<std::vector<double>> mY(4, std::vector<double>(6));
    
    for (int i = 0; i < 6; ++i) {
        // Note: We're swapping X and Y for the inverse curve
        mY[0][i] = concPts1[i][0];
        mY[1][i] = concPts2[i][0];
        mY[2][i] = eccPts1[i][0];
        mY[3][i] = eccPts2[i][0];
        
        mX[0][i] = concPts1[i][1];
        mX[1][i] = concPts2[i][1];
        mX[2][i] = eccPts1[i][1];
        mX[3][i] = eccPts2[i][1];
    }
    
    return SmoothSegmentedFunction(
        mY, mX,
        yC, yE,
        xC, xE,
        1 / dydxC, 1 / dydxE,
        computeIntegral,
        true
    );
}

SmoothSegmentedFunction SmoothSegmentedFunction::createFiberActiveForceLengthCurve(
    double x0, double x1, double x2, double x3, double ylow,
    double dydx, double curviness, bool computeIntegral)
{
    // Ensure that the inputs are within a valid range
    double root_eps = std::sqrt(std::numeric_limits<double>::epsilon());
    assert(x0 >= 0 && x1 > x0 + root_eps && x2 > x1 + root_eps && x3 > x2 + root_eps && "This must be true: 0 < lce0 < lce1 < lce2 < lce3");
    assert(ylow >= 0 && "shoulderVal must be greater than, or equal to 0");
    double dydx_upper_bound = (1 - ylow) / (x2 - x1);
    assert(0 <= dydx && dydx < dydx_upper_bound && "plateauSlope must be greater than 0 and less than the upper bound");
    assert(0 <= curviness && curviness <= 1 && "curviness must be between 0 and 1");
    
    // Translate the users parameters into Bezier curves
    double c = scaleCurviness(curviness);
    
    // The active force length curve is made up of 5 elbow shaped sections.
    // Compute the locations of the joining point of each elbow section.
    
    // Calculate the location of the shoulder
    double xDelta = 0.05 * x2;  // half the width of the sarcomere
    
    double xs = (x2 - xDelta);  // x1 + 0.75*(x2-x1)
    
    // Calculate the intermediate points located on the ascending limb
    double y0 = 0, dydx0 = 0;
    double y1 = 1 - dydx * (xs - x1);
    double dydx01 = 1.25 * (y1 - y0) / (x1 - x0);
    
    double x01 = x0 + 0.5 * (x1 - x0);
    double y01 = y0 + 0.5 * (y1 - y0);
    
    // Calculate the intermediate points of the shallow ascending plateau
    double x1s = x1 + 0.5 * (xs - x1);
    double y1s = y1 + 0.5 * (1 - y1);
    double dydx1s = dydx;
    
    // x2 entered
    double y2 = 1, dydx2 = 0;
    
    // Descending limb
    // x3 entered
    double y3 = 0, dydx3 = 0;
    
    double x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta));
    double y23 = y2 + 0.5 * (y3 - y2);
    double dydx23 = (y3 - y2) / ((x3 - xDelta) - (x2 + xDelta));
    
    // Compute the locations of the control points
    auto p0 = QuinticBezierCurve::calcCornerControlPoints(x0, ylow, dydx0, x01, y01, dydx01, c);
    auto p1 = QuinticBezierCurve::calcCornerControlPoints(x01, y01, dydx01, x1s, y1s, dydx1s, c);
    auto p2 = QuinticBezierCurve::calcCornerControlPoints(x1s, y1s, dydx1s, x2, y2, dydx2, c);
    auto p3 = QuinticBezierCurve::calcCornerControlPoints(x2, y2, dydx2, x23, y23, dydx23, c);
    auto p4 = QuinticBezierCurve::calcCornerControlPoints(x23, y23, dydx23, x3, ylow, dydx3, c);
    
    std::vector<std::vector<double>> mX(5, std::vector<double>(6));
    std::vector<std::vector<double>> mY(5, std::vector<double>(6));
    
    for (int i = 0; i < 6; ++i) {
        mX[0][i] = p0[i][0];
        mX[1][i] = p1[i][0];
        mX[2][i] = p2[i][0];
        mX[3][i] = p3[i][0];
        mX[4][i] = p4[i][0];
        
        mY[0][i] = p0[i][1];
        mY[1][i] = p1[i][1];
        mY[2][i] = p2[i][1];
        mY[3][i] = p3[i][1];
        mY[4][i] = p4[i][1];
    }
    
    return SmoothSegmentedFunction(
        mX, mY,
        x0, x3,
        ylow, ylow,
        0, 0,
        computeIntegral,
        true
    );
}