import numpy as np
from scipy.optimize import fsolve
from CurveActiveForceLength import CurveActiveForceLength
from CurveFiberForceLength import CurveFiberForceLength
from CurveTendonForceLength import CurveTendonForceLength
from CurveForceVelocity import CurveForceVelocity

# Computes vM from lM and lMT
def calcVel(lM, lMT, act, params, curves):
    lMtilde = lM / params['lMopt']
    alphaM = calcPennationAngleTilde(lMtilde, params)
    lT = lMT - lM * np.cos(alphaM)
    lTtilde = lT / params['lTslack']
    vMtilde = calcVelTilde(lMtilde, lTtilde, act, params, curves)
    vM = vMtilde * params['lMopt'] * params['vMmax']
    if vM < 0 and lM < params['lMmin']:
        vM = 0
    return vM

# Computes vMtilde from lMtilde and lTtilde
# This is the function we want to fit to.
def calcVelTilde(lMtilde, lTtilde, act, params, curves):
    alphaM = calcPennationAngleTilde(lMtilde, params)
    cosAlphaM = np.cos(alphaM)
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    tfl = curves['TFL'].calcValue(lTtilde)
    vMtildeInit = 0

    # Use fsolve to compute the muscle fiber velocity
    vMtilde = fsolve(lambda vMtilde: forceBalance(vMtilde, act, afl, pfl, tfl, curves['FV'], cosAlphaM, params), vMtildeInit)[0]
    return vMtilde

# Computes the pennation angle from lM
def calcPennationAngle(lM, params):
    if params['alphaMopt'] > np.finfo(float).eps:
        if lM > params['lMmin']:
            sinAlpha = params['h'] / lM
            if sinAlpha < params['alphaMax']:
                alphaM = np.arcsin(sinAlpha)
            else:
                alphaM = params['alphaMax']
        else:
            alphaM = params['alphaMax']
    else:
        alphaM = 0
    return alphaM

# Computes the pennation angle from lMtilde
def calcPennationAngleTilde(lMtilde, params):
    if params['alphaMopt'] > np.finfo(float).eps:
        htilde = params['h'] / params['lMopt']
        if lMtilde > params['lMmin'] / params['lMopt']:
            sinAlpha = htilde / lMtilde
            if sinAlpha < params['alphaMax']:
                alphaM = np.arcsin(sinAlpha)
            else:
                alphaM = params['alphaMax']
        else:
            alphaM = params['alphaMax']
    else:
        alphaM = 0
    return alphaM

# Force balance
# This function computes the current force discrepancy between
# the muscle and the tendon, which we want to be zero
def forceBalance(vMtilde, act, afl, pfl, tfl, curveFV, cosAlphaM, params):
    deriv = curveFV.calcValDeriv(vMtilde)
    fv = deriv[0]
    dfv = deriv[1]
    fM = act * afl * fv + pfl + params['beta'] * vMtilde
    f = fM * cosAlphaM - tfl
    J = (act * afl * dfv + params['beta']) * cosAlphaM
    return f

# Muscle parameters
params = {
    'beta': 0.1,  # damping
    'lMopt': 0.1,  # optimal muscle length
    'lTslack': 0.2,  # tendon slack length
    'vMmax': 10,  # maximum contraction velocity
    'alphaMopt': np.pi / 6,  # pennation angle at optimal muscle length
    'fMopt': 100,  # peak isometric force
    'amin': 0.01,  # minimum activation
    'tauA': 0.01,  # activation constant
    'tauD': 0.4,  # deactivation constant
}

# Muscle curves
curves = {
    'AFL': CurveActiveForceLength(),
    'PFL': CurveFiberForceLength(),
    'TFL': CurveTendonForceLength(),
    'FV': CurveForceVelocity()
}

# Initialize
params['alphaMax'] = np.arccos(0.1)
params['h'] = params['lMopt'] * np.sin(params['alphaMopt'])
params['lMT'] = params['lTslack'] + params['lMopt'] * np.cos(params['alphaMopt'])

if params['alphaMax'] > 1e-6:
    minPennatedFiberLength = params['h'] / np.sin(params['alphaMax'])
else:
    minPennatedFiberLength = params['lMopt'] * 0.01

minActiveFiberLength = curves['AFL'].min_norm_active_fiber_length * params['lMopt']
params['lMmin'] = max(minActiveFiberLength, minPennatedFiberLength)

# Test: initial equilibrium
# Computes the equilibrium fiber length, given the initial
# activation and the initial musculotendon length.
# Given lM, we can compute vM that gives force balance. This vM
# may or may not be zero. For initialization, we want this vM
# to be zero.
# lM = fsolve(lambda lM: calcVel(lM, lMT, act, params, curves), lM)[0]
# lMtilde = lM / params['lMopt']
# lb = params['lMmin'] / params['lMopt']
# ub = 2.0
# lMtilde = max(lb, min(lMtilde, ub))
# print(f'Initial lMtilde: {lMtilde:.6f}')