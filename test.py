import numpy as np
import pandas as pd
import sys
sys.path.append('../python-nn')
from testMuscle import *

"""
sample_size = 100
act = 1
lMtilde = np.linspace(.30, 2.00, sample_size)
lTtilde = np.linspace(.985, 1.075, sample_size)
X, Y = np.meshgrid(lMtilde, lTtilde)

vMtilde = []
lm_r = []
lt_r = []
for x, y in zip(X.ravel(), Y.ravel()):
    lm_r.append(x)
    lt_r.append(y)
    vMtilde.append(calcVelTilde(x, y, act, params, curves))

DF = pd.DataFrame(np.column_stack([lm_r, lt_r, vMtilde]), 
                               columns=['lMtilde', 'lTtilde', 'vMtilde'])
DF.to_csv(f"twoDimensional/velocityData_2D_a={act}.csv")
"""

x = [1,2,3,4,5]
y = [1,1,1,1,1]
# Organize the prediction such that it has the exact shape as ground truth
DF = pd.DataFrame(np.column_stack([x, y]), 
                               columns=['x', 'y'])
DF.loc[(DF.x > 2) & (DF.x < 4), 'y'] = np.nan
#DF.loc[DF.x < 4, 'y'] = np.nan
print(DF)