#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "Muscle.h"

// Example program that demonstrates using the muscle model
// for various calculations

void printMuscleForce(const Muscle& muscle, double lM, double lMT, 
                     double act, bool computeVelocity) {
    // Calculate pennation angle
    double alphaM = muscle.calcPennationAngle(lM);
    
    // If we need to compute velocity from force equilibrium
    double vM = 0.0;
    if (computeVelocity) {
        vM = muscle.computeVel(lM, lMT, act, alphaM);
        std::cout << "Computed vM: " << vM << " m/s" << std::endl;
    } else {
        // Use a fixed velocity
        vM = -0.1; // Shortening velocity
        std::cout << "Fixed vM: " << vM << " m/s" << std::endl;
    }
    
    // Compute muscle force
    double force = muscle.muscleForce(vM, lM, act, alphaM);
    
    // Output results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Muscle state:" << std::endl;
    std::cout << "  Fiber length (lM): " << lM << " m" << std::endl;
    std::cout << "  MT length (lMT): " << lMT << " m" << std::endl;
    std::cout << "  Activation (act): " << act << std::endl;
    std::cout << "  Pennation angle: " << alphaM << " rad (" 
              << alphaM * 180.0 / M_PI << " deg)" << std::endl;
    std::cout << "  Fiber velocity (vM): " << vM << " m/s" << std::endl;
    std::cout << "  Muscle force: " << force << " N" << std::endl;
    std::cout << std::endl;
}

void runLengthSweep(const Muscle& muscle) {
    std::cout << "===== MUSCLE LENGTH SWEEP =====" << std::endl;
    
    const int numPoints = 11;
    const double lMopt = 0.1; // optimal fiber length
    const double lMmin = 0.7 * lMopt;
    const double lMmax = 1.3 * lMopt;
    const double lMstep = (lMmax - lMmin) / (numPoints - 1);
    
    const double lMT = 0.3; // fixed MT length
    const double act = 0.8; // fixed activation
    
    std::cout << "lM (m)\tForce (N)" << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (int i = 0; i < numPoints; ++i) {
        double lM = lMmin + i * lMstep;
        double alphaM = muscle.calcPennationAngle(lM);
        
        // Compute velocity from force equilibrium
        double vM = muscle.computeVel(lM, lMT, act, alphaM);
        
        // Compute force
        double force = muscle.muscleForce(vM, lM, act, alphaM);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << lM << "\t" << force << std::endl;
    }
    std::cout << std::endl;
}

void runActivationSweep(const Muscle& muscle) {
    std::cout << "===== ACTIVATION SWEEP =====" << std::endl;
    
    const int numPoints = 11;
    const double actMin = 0.0;
    const double actMax = 1.0;
    const double actStep = (actMax - actMin) / (numPoints - 1);
    
    const double lM = 0.1; // optimal fiber length
    const double lMT = 0.3; // fixed MT length
    
    std::cout << "Activation\tForce (N)" << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (int i = 0; i < numPoints; ++i) {
        double act = actMin + i * actStep;
        double alphaM = muscle.calcPennationAngle(lM);
        
        // Compute velocity from force equilibrium
        double vM = muscle.computeVel(lM, lMT, act, alphaM);
        
        // Compute force
        double force = muscle.muscleForce(vM, lM, act, alphaM);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << act << "\t" << force << std::endl;
    }
    std::cout << std::endl;
}

void runVelocitySweep(const Muscle& muscle) {
    std::cout << "===== VELOCITY SWEEP =====" << std::endl;
    
    const int numPoints = 21;
    const double vMmin = -1.0; // normalized velocity (shortening)
    const double vMmax = 1.0;  // normalized velocity (lengthening)
    const double vMstep = (vMmax - vMmin) / (numPoints - 1);
    
    const double lM = 0.1; // optimal fiber length
    const double act = 0.8; // fixed activation
    const double alphaM = muscle.calcPennationAngle(lM);
    
    std::cout << "vM (norm)\tForce (N)" << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (int i = 0; i < numPoints; ++i) {
        double vM_norm = vMmin + i * vMstep;
        double vM = vM_norm * 0.1 * 10.0; // Convert to m/s using lMopt * vMmax
        
        // Compute force
        double force = muscle.muscleForce(vM, lM, act, alphaM);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << vM_norm << "\t" << force << std::endl;
    }
    std::cout << std::endl;
}

void runPerformanceTest(const Muscle& muscle) {
    std::cout << "===== PERFORMANCE TEST =====" << std::endl;
    
    // Input data size
    const int dataSize = 10000;
    
    // Create input data vectors
    std::vector<double> vM(dataSize), lM(dataSize), act(dataSize), alphaM(dataSize);
    
    // Fill with sample data
    for (int i = 0; i < dataSize; ++i) {
        vM[i] = (i % 20 - 10) * 0.01;                      // Velocities between -0.1 and 0.1
        lM[i] = 0.08 + (i % 40) * 0.001;                   // Lengths around optimal length
        act[i] = 0.2 + (i % 9) * 0.1;                      // Activations between 0.2 and 1.0
        alphaM[i] = muscle.calcPennationAngle(lM[i]);      // Pennation angles based on length
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> forces = muscle.muscleForceVectorized(vM, lM, act, alphaM);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "Computed " << dataSize << " muscle forces in " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per force computation: " << elapsed.count() / dataSize << " ms" << std::endl;
    std::cout << "Performance: " << (dataSize / elapsed.count() * 1000) << " forces/second" << std::endl;
    std::cout << std::endl;
}

int main() {
    // Create muscle with default parameters
    Muscle muscle(0.1,    // lMopt - optimal muscle length
                  0.2,    // lTslack - tendon slack length
                  10.0,   // vMmax - maximum contraction velocity
                  M_PI/6, // alphaMopt - pennation angle at optimal fiber length
                  100.0); // fMopt - peak isometric force
    
    std::cout << "===== MUSCLE FORCE CALCULATOR EXAMPLE =====" << std::endl << std::endl;
    
    // Example 1: Calculate muscle force at specific state
    std::cout << "Example 1: Specific muscle state" << std::endl;
    printMuscleForce(muscle, 0.09, 0.28, 0.8, true);
    
    // Example 2: Calculate muscle force with fixed velocity
    std::cout << "Example 2: Fixed velocity" << std::endl;
    printMuscleForce(muscle, 0.09, 0.28, 0.8, false);
    
    // Example 3: Length sweep
    runLengthSweep(muscle);
    
    // Example 4: Activation sweep
    runActivationSweep(muscle);
    
    // Example 5: Velocity sweep
    runVelocitySweep(muscle);
    
    // Example 6: Performance test
    runPerformanceTest(muscle);
    
    return 0;
}