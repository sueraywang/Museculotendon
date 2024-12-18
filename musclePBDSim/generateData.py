import numpy as np
import pandas as pd
from Physics import compute_spring_force
import sys
sys.path.append('../Museculotendon')
from testMuscle import *
from Physics import *
import matplotlib.pyplot as plt

# Parameters
m = MASS
k = SPRING_CONSTANT 
d = DAMPING_CONSTANT
x0 = CLASSIC_FREE_POS[1]
v0 = 0.0

# Time settings
dt = DT/SUB_STEPS           # Time step (s)
t_max = 10          # Total simulation time (s)

# Initialize time, positions, and velocities
t = [0]
x = [x0]
v = [v0]

t_now = 0
i = 0
# Simulation loop
while t_now < t_max:
    # Compute forces
    spring_force = -k * (x[i]-REST_LENGTH) - d * v[i]

    # Update
    a = spring_force / m
    v_next = v[i] + a * dt
    x_next = x[i] + v_next * dt
    t_now = t[i] + dt
    i += 1
    v.append(v_next)
    x.append(x_next)
    t.append(t_now)

dx = []
dv = []
for i in x:
    dx.append(i-REST_LENGTH)
for i in v:
    dv.append(DAMPING_CONSTANT / SPRING_CONSTANT * i)
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

