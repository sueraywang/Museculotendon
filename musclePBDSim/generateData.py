import numpy as np
import pandas as pd
from Physics import compute_spring_force
import sys
sys.path.append('../Museculotendon')
from testMuscle import *
from Physics import *
import matplotlib.pyplot as plt
    
"""
DF = pd.DataFrame(np.column_stack([lMtilde, force]), 
                               columns=['lMtilde', 'force'])
DF.to_csv("musclePBDSim/lM_foce_data.csv")
"""

# Parameters
m1, m2 = MASS, MASS
k = SPRING_CONSTANT 
c = DAMPING_CONSTANT
x1_0, x2_0 = CLASSIC_FIX_POS[1], CLASSIC_FREE_POS[1]
v1_0, v2_0 = 0.0, 0.0   # Initial velocities

# Time settings
dt = 0.001           # Time step (s)
t_max = 10          # Total simulation time (s)

# Initialize time, positions, and velocities
t = np.arange(0, t_max, dt)
x1 = np.zeros_like(t)
x2 = np.zeros_like(t)
v1 = np.zeros_like(t)
v2 = np.zeros_like(t)
dx = np.zeros_like(t)
dv = np.zeros_like(t)

# Set initial conditions
x1[0] = x1_0
x2[0] = x2_0
v1[0] = v1_0
v2[0] = v2_0
dx[0] = x1[0] - x2[0] - REST_LENGTH
dv[0] = v1[0] - v2[0]

# Simulation loop
for i in range(1, len(t)):
    # Compute forces
    spring_force = -k * (x1[i-1] - x2[i-1]-REST_LENGTH)
    damping_force = -c * (v1[i-1] - v2[i-1])

    # Update accelerations
    a1 = (spring_force + damping_force) / m1
    a2 = (-spring_force - damping_force) / m2

    # Update velocities and positions using Euler's method
    v1[i] = v1[i-1] + a1 * dt
    v2[i] = v2[i-1] + a2 * dt
    x1[i] = x1[i-1] + v1[i-1] * dt
    x2[i] = x2[i-1] + v2[i-1] * dt
    dx[i] = x1[i] - x2[i] - REST_LENGTH
    dv[i] = v1[i] - v2[i]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, dx, label='Particle 1 (x1)')
plt.title("Two-Particle System")
plt.xlabel("Time (s)")
plt.ylabel("Length (m)")
plt.legend()
plt.grid()
plt.show()


DF = pd.DataFrame(np.column_stack([dx, dv]), 
                               columns=['dx', 'dv'])
DF.to_csv("musclePBDSim/dampedLinearSpringData.csv")


