#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include "Muscle.h"

// Example usage and performance test
int main() {
    // Create muscle with default parameters
    Muscle muscle(0.1,    // lMopt
                  0.2,    // lTslack
                  10,     // vMmax
                  M_PI/6, // alphaMopt
                  1.0);   // fMopt
    
    std::cout << "Muscle model initialized" << std::endl;
    
    // Input data size
    const int dataSize = 1000;
    
    // Create input data vectors
    std::vector<double> vM(dataSize), lM(dataSize), act(dataSize), alphaM(dataSize);
    
    // Fill with sample data
    for (int i = 0; i < dataSize; ++i) {
        vM[i] = (i % 20 - 10) * 0.1;                       // Velocities between -1 and 1
        lM[i] = 0.08 + (i % 40) * 0.001;                   // Lengths around optimal length
        act[i] = 0.2 + (i % 9) * 0.1;                      // Activations between 0.2 and 1.0
        alphaM[i] = muscle.calcPennationAngle(lM[i]);      // Pennation angles based on length
    }
    
    // Test individual force computation
    double testForce = muscle.muscleForce(vM[0], lM[0], act[0], alphaM[0]);
    std::cout << "Single force test: " << testForce << std::endl;
    
    // Measure performance of vectorized computation
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> forces = muscle.muscleForceVectorized(vM, lM, act, alphaM);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "Computed " << dataSize << " muscle forces in " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per force computation: " << elapsed.count() / dataSize << " ms" << std::endl;
    
    // Output first few values for verification
    std::cout << "\nSample output (first 5 values):" << std::endl;
    for (int i = 0; i < 5 && i < dataSize; ++i) {
        std::cout << "vM=" << vM[i] << ", lM=" << lM[i] 
                  << ", act=" << act[i] << ", alphaM=" << alphaM[i] 
                  << " -> Force=" << forces[i] << std::endl;
    }
    
    return 0;
}