import numpy as np
import pandas as pd
import random
import sys
sys.path.append('../Museculotendon')
from testMuscle import *

sample_size = 10000
fixed_a = 1

lMtilde = np.linspace(.30, 2.00, sample_size)

def muscleForce(lMtilde, act, params):
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    fM = act * afl + pfl
    f = fM * params['fMopt']
    return f

force = muscleForce(lMtilde, fixed_a, params)

DF = pd.DataFrame(np.column_stack([lMtilde, force]), 
                               columns=['lMtilde', 'force'])
DF.to_csv("musclePBDSim/lM_foce_data.csv")