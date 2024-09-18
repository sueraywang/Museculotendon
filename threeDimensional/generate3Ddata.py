import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../python-nn')
from testMuscle import *

sample_size = 100000
fixed_a = 1
vRange = 1.75

lMtilde = list(np.random.uniform(.30, 2.00, sample_size))
lTtilde = list(np.random.uniform(.985, 1.075, sample_size))
act = list(np.random.uniform(0, 1, sample_size))

r_lM = []
r_lT = []
r_act = []
vMtilde = []

# Evaluate the function on the grid
while (len(vMtilde) < 10000):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = random.choice(act)
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 11000):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 0
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 12000):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 1
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

"""
while (len(vMtilde) < 11500):
    lM = .40
    lT = random.choice(lTtilde)
    a = fixed_a
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= vRange) & (v >= -vRange)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

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
DF.to_csv("random3DVelocityData.csv")