#include "Muscle.h"

Muscle::Muscle(double lMopt, double lTslack, double vMmax, double alphaMopt, double fMopt, 
               double beta, double amin, double tauA, double tauD) 
    : beta_(beta), lMopt_(lMopt), lTslack_(lTslack), vMmax_(vMmax), alphaMopt_(alphaMopt),
      fMopt_(fMopt), amin_(amin), tauA_(tauA), tauD_(tauD) {
    initialize();
}

void Muscle::initialize() {
    // Initialize derived parameters
    alphaMax_ = std::acos(0.1);
    h_ = lMopt_ * std::sin(alphaMopt_);
    lMT_ = lTslack_ + lMopt_ * std::cos(alphaMopt_);
    
    double minPennatedFiberLength;
    if (alphaMax_ > 1e-6) {
        minPennatedFiberLength = h_ / std::sin(alphaMax_);
    } else {
        minPennatedFiberLength = lMopt_ * 0.01;
    }
    
    double minActiveFiberLength = curveAFL_.getMinNormActiveFiberLength() * lMopt_;
    lMmin_ = std::max(minActiveFiberLength, minPennatedFiberLength);
}

double Muscle::calcPennationAngle(double lM) const {
    double alphaM = 0.0;
    
    if (alphaMopt_ > std::numeric_limits<double>::epsilon()) {
        if (lM > lMmin_) {
            double sinAlpha = h_ / lM;
            if (sinAlpha < alphaMax_) {
                alphaM = std::asin(sinAlpha);
            } else {
                alphaM = alphaMax_;
            }
        } else {
            alphaM = alphaMax_;
        }
    }
    
    return alphaM;
}

double Muscle::calcPennationAngleTilde(double lMtilde) const {
    double alphaM = 0.0;
    
    if (alphaMopt_ > std::numeric_limits<double>::epsilon()) {
        double htilde = h_ / lMopt_;
        if (lMtilde > lMmin_ / lMopt_) {
            double sinAlpha = htilde / lMtilde;
            if (sinAlpha < alphaMax_) {
                alphaM = std::asin(sinAlpha);
            } else {
                alphaM = alphaMax_;
            }
        } else {
            alphaM = alphaMax_;
        }
    }
    
    return alphaM;
}

std::pair<double, double> Muscle::forceBalance(double vMtilde, double act, double afl, double pfl, 
                                             double tfl, const CurveForceVelocity& curveFV, 
                                             double cosAlphaM) const {
    std::vector<double> deriv = curveFV.calcValDeriv(vMtilde);
    double fv = deriv[0];
    double dfv = deriv[1];
    
    double fM = act * afl * fv + pfl + beta_ * vMtilde;
    double f = fM * cosAlphaM - tfl;
    double J = (act * afl * dfv + beta_) * cosAlphaM;
    
    return std::make_pair(f, J);
}

double Muscle::computeVelTilde(double lMtilde, double lTtilde, double act, double alphaM) const {
    double cosAlphaM = std::cos(alphaM);
    double afl = curveAFL_.calcValue(lMtilde);
    double pfl = curvePFL_.calcValue(lMtilde);
    double tfl = curveTFL_.calcValue(lTtilde);
    double vMtildeInit = 0.0;
    
    // Define the objective function for Newton solver
    auto objective = [this, act, afl, pfl, tfl, cosAlphaM](double vMtilde) -> std::pair<double, double> {
        return this->forceBalance(vMtilde, act, afl, pfl, tfl, this->curveFV_, cosAlphaM);
    };
    
    // Use Newton solver to find the muscle fiber velocity
    NewtonSolver solver;
    double vMtilde = solver.solve(objective, vMtildeInit);
    
    return vMtilde;
}

double Muscle::computeVel(double lM, double lMT, double act, double alphaM) const {
    double lMtilde = lM / lMopt_;
    double lT = lMT - lM * std::cos(alphaM);
    double lTtilde = lT / lTslack_;
    
    double vMtilde = computeVelTilde(lMtilde, lTtilde, act, alphaM);
    double vM = vMtilde * lMopt_ * vMmax_;
    
    return vM;
}

double Muscle::muscleForce(double vM, double lM, double act, double alphaM) const {
    double lMtilde = lM / lMopt_;
    double vMtilde = vM / (lMopt_ * vMmax_);
    
    double afl = curveAFL_.calcValue(lMtilde);
    double pfl = curvePFL_.calcValue(lMtilde);
    std::vector<double> deriv = curveFV_.calcValDeriv(vMtilde);
    double fv = deriv[0];
    
    double fM = act * afl * fv + pfl + beta_ * vMtilde;
    double f = fM * std::cos(alphaM) * fMopt_;
    
    return f;
}

std::vector<double> Muscle::muscleForceVectorized(const std::vector<double>& vM, 
                                                const std::vector<double>& lM,
                                                const std::vector<double>& act, 
                                                const std::vector<double>& alphaM) const {
    size_t n = vM.size();
    std::vector<double> forces(n);
    
    for (size_t i = 0; i < n; ++i) {
        forces[i] = muscleForce(vM[i], lM[i], act[i], alphaM[i]);
    }
    
    return forces;
}