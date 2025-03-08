#ifndef MUSCLE_H
#define MUSCLE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "SmoothSegmentedFunction.h"
#include "CurveActiveForceLength.h"
#include "CurveFiberForceLength.h"
#include "CurveForceVelocity.h"
#include "CurveTendonForceLength.h"
#include "NewtonSolver.h"

class Muscle {
public:
    // Constructor
    Muscle(double lMopt, double lTslack, double vMmax, double alphaMopt, double fMopt, 
           double beta = 0.1, double amin = 0.01, double tauA = 0.01, double tauD = 0.4);
    
    // Initialize the muscle model
    void initialize();
    
    // Core computation functions
    double calcPennationAngle(double lM) const;
    double calcPennationAngleTilde(double lMtilde) const;
    
    // Force balance function for Newton solver
    std::pair<double, double> forceBalance(double vMtilde, double act, double afl, double pfl, 
                                         double tfl, const CurveForceVelocity& curveFV, 
                                         double cosAlphaM) const;
    
    // Velocity computations
    double computeVelTilde(double lMtilde, double lTtilde, double act, double alphaM) const;
    double computeVel(double lM, double lMT, double act, double alphaM) const;
    
    // Muscle force computation
    double muscleForce(double vM, double lM, double act, double alphaM) const;
    
    // Vectorized muscle force computation for performance
    std::vector<double> muscleForceVectorized(const std::vector<double>& vM, 
                                            const std::vector<double>& lM,
                                            const std::vector<double>& act, 
                                            const std::vector<double>& alphaM) const;
    
private:
    // Muscle parameters
    double beta_;       // Damping
    double lMopt_;      // Optimal muscle length
    double lTslack_;    // Tendon slack length
    double vMmax_;      // Maximum contraction velocity
    double alphaMopt_;  // Pennation angle at optimal muscle length
    double fMopt_;      // Peak isometric force
    double amin_;       // Minimum activation
    double tauA_;       // Activation constant
    double tauD_;       // Deactivation constant
    
    // Derived parameters
    double alphaMax_;   // Maximum pennation angle
    double h_;          // Height
    double lMT_;        // Musculotendon length
    double lMmin_;      // Minimum muscle length
    
    // Muscle curves
    CurveActiveForceLength curveAFL_;
    CurveFiberForceLength curvePFL_;
    CurveTendonForceLength curveTFL_;
    CurveForceVelocity curveFV_;
};

#endif // MUSCLE_H