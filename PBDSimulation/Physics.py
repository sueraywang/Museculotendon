# Physics.py
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('ComputeMuscleForce'))
from CurveActiveForceLength import CurveActiveForceLength
from CurveFiberForceLength import CurveFiberForceLength
from CurveTendonForceLength import CurveTendonForceLength
from CurveForceVelocity import CurveForceVelocity

# Muscle parameters
params = {
    'beta': 0.1,  # damping
    'lMopt': 0.1,  # optimal muscle length
    'lTslack': 0.2,  # tendon slack length
    'vMmax': 10,  # maximum contraction velocity
    'alphaMopt': np.pi / 6,  # pennation angle at optimal muscle length
    'fMopt': 1,  # peak isometric force, originally = 100
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

# Physical constants
MASS = 0.1  # Mass of particle (kg)
GRAVITY = np.array([0, -9.8, 0])

# Spring parameters
REST_LENGTH = 1.0
SPRING_CONSTANT = 10  # Spring constant
DAMPING_CONSTANT = 0.1
SPRING_COMPLIANCE = 1/SPRING_CONSTANT

# XPBD constants
DT = 0.01  # FPS = 60
SUB_STEPS = 100
MUSCLE_COMPLIANCE = 1/params['fMopt']

# Objects' initial status
INITIAL_LENGTH = 0.1
XPBD_FIX_POS = np.array([0.5, 0.5, 0.0])
CLASSIC_FIX_POS = np.array([-0.5, 0.5, 0.0])
XPBD_FREE_POS = XPBD_FIX_POS + np.array([0.0, -INITIAL_LENGTH, 0.0])
CLASSIC_FREE_POS = CLASSIC_FIX_POS + np.array([0.0, -INITIAL_LENGTH, 0.0])

# Helper functions
def normalized(vec):
    return vec / np.linalg.norm(vec)

def compute_spring_force(pos1, pos2, k, rest_len):
    displacement = pos1 - pos2
    length = np.linalg.norm(displacement)
    if length == 0:
        return np.array([0.0, 0.0, 0.0])
    force_magnitude = -k * (length - rest_len)**3
    return force_magnitude * normalized(displacement)

def muscleForce(lMtilde, act=1.0, pennation = 0.0):
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    fM = act * afl + pfl
    return fM * np.cos(np.radians(pennation))

class Particle:
    def __init__(self, position, mass=MASS, fixed=False, xpbd=False):
        self.position = position
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(3)
        self.prev_velocity = self.velocity.copy()
        self.mass = mass
        self.weight = 1.0 / mass if mass != 0.0 else 0.0
        self.fixed = fixed
        self.xpbd = xpbd

class Constraint:
    def __init__(self, p1, p2, compliance):
        self.p1 = p1
        self.p2 = p2
        self.compliance = compliance
        self.lambda_acc = 0.0