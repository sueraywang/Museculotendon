# Muscle Force Calculator

A high-performance C++ implementation of muscle force calculations based on the Hill-type muscle model.

## Overview

This project provides a C++ implementation of the Hill-type muscle model, converting Python code to C++ for improved performance. The implementation includes:

- Smooth segmented functions for muscle-tendon dynamics
- Active force-length relationship
- Passive force-length relationship
- Force-velocity relationship
- Tendon force-length relationship
- Pennation angle calculations
- Force equilibrium solvers

## Project Structure

- `Muscle.h/cpp`: Main muscle model class
- `CurveBase.h`: Base class for all muscle curves
- `CurveActiveForceLength.h/cpp`: Active force-length relationship
- `CurveFiberForceLength.h/cpp`: Passive force-length relationship 
- `CurveForceVelocity.h/cpp`: Force-velocity relationship
- `CurveTendonForceLength.h/cpp`: Tendon force-length relationship
- `SmoothSegmentedFunction.h/cpp`: Core curve implementation using quintic Bezier splines
- `QuinticBezierCurve.h/cpp`: Bezier curve mathematics
- `NewtonSolver.h`: Newton-Raphson solver for force equilibrium
- `main.cpp`: Example usage and performance test

## Performance

This C++ implementation is significantly faster than the equivalent Python implementation, especially for large batch computations. The vectorized interface allows for efficient calculation of many muscle forces in one call, perfect for simulations or large-scale analyses.

## How to Build

Simply use the provided Makefile:

```bash
make
```

This will build the executable `muscle_force_calculator`.

To run the performance test:

```bash
./muscle_force_calculator
```

This will calculate 1,000 muscle forces and report the computation time.

## Installation

### Prerequisites

- C++14 compatible compiler (GCC, Clang, MSVC)
- Make build system (optional, you can compile manually)

### Integrating into Your Project

You can either:

1. Include the source files directly in your project
2. Compile as a static library and link against it
3. Use the provided Makefile to build a standalone executable

## Usage Example

```cpp
// Create a muscle with default parameters
Muscle muscle(0.1,    // lMopt - optimal muscle length
              0.2,    // lTslack - tendon slack length
              10,     // vMmax - maximum contraction velocity
              M_PI/6, // alphaMopt - pennation angle at optimal length
              1.0);   // fMopt - peak isometric force

// Calculate muscle force for a given state
double vM = -0.1;      // normalized fiber velocity
double lM = 0.09;      // fiber length
double act = 0.8;      // activation level
double alphaM = muscle.calcPennationAngle(lM);  // pennation angle

// Compute force
double force = muscle.muscleForce(vM, lM, act, alphaM);

// For batch computations, use the vectorized interface for better performance
std::vector<double> vM_batch = {-0.1, -0.05, 0.0, 0.05, 0.1};
std::vector<double> lM_batch = {0.09, 0.095, 0.1, 0.105, 0.11};
std::vector<double> act_batch = {0.8, 0.7, 0.6, 0.5, 0.4};
std::vector<double> alphaM_batch(5);

// Calculate pennation angles
for (size_t i = 0; i < lM_batch.size(); ++i) {
    alphaM_batch[i] = muscle.calcPennationAngle(lM_batch[i]);
}

// Compute forces in batch
std::vector<double> forces = muscle.muscleForceVectorized(vM_batch, lM_batch, act_batch, alphaM_batch);
```