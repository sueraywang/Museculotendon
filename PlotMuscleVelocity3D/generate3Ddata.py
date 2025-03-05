import numpy as np
import pandas as pd
import random
import sys
import os
sys.path.append(os.path.abspath('ComputeMuscleForce'))
from testMuscle import *

sample_size = 100000
data_size = 5e4
boundary_size = 5e3
fixed_a = 1
vRange = 1.75

lMtilde = list(np.random.uniform(.30, 2.00, sample_size))
lTtilde = list(np.random.uniform(.985, 1.075, sample_size))
act = list(np.random.uniform(0, 1, sample_size))
# Special sampling for boundary data
lMtilde_b = list(np.random.uniform(1.002, 1.006, int(sample_size * 0.004/(1.075-0.985))))
lTtilde_b = list(np.random.uniform(.3, .5, int(sample_size * 0.2/(2.00-0.30))))

r_lM = []
r_lT = []
r_act = []
vMtilde = []

# Evaluate the function on the grid
while (len(vMtilde) < data_size):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = random.choice(act)
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < data_size + boundary_size):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 0
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < data_size + boundary_size*2):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 1
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)


while (len(vMtilde) < data_size + boundary_size*4):
    lM = random.choice(lMtilde_b)
    lT = random.choice(lTtilde_b)
    a = fixed_a
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

"""
vMtilde.append(calcVelTilde(0.471717, 1.003182, fixed_a, params, curves))
vMtilde.append(calcVelTilde(0.3, 1.003108, fixed_a, params, curves))
r_lM.append(0.471717)
r_lM.append(0.3)
r_lT.append(1.003182)
r_lT.append(1.003108)
r_act.append(fixed_a)
r_act.append(fixed_a)

while (len(vMtilde) < 12000):
    lM = 1.90
    lT = random.choice(lTtilde)
    a = fixed_a
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 12500):
    lM = random.choice(lMtilde)
    lT = 0.99
    a = fixed_a
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 13000):
    lM = random.choice(lMtilde)
    lT = 1.07
    a = fixed_a
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)
"""

DF = pd.DataFrame(np.column_stack([r_lM, r_lT, r_act, vMtilde]), 
                               columns=['lMtilde', 'lTtilde', 'act', 'vMtilde'])
DF.to_csv("PlotMuscleVelocity3D/mini_random3DVelocityData.csv")