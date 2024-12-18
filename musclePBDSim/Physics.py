# Physics.py
# Objects/Constrains are created here
import numpy as np

# Physical constants
MASS = 0.1  # Mass of particle (kg)
GRAVITY = np.array([0, -9.8, 0])
REST_LENGTH = 1.0
SPRING_CONSTANT = 10  # Spring constant
DT = 1/60  # FPS = 60

# XPBD constants
SUB_STEPS = 500
COMPLIANCE = 1/SPRING_CONSTANT

# Objects' initial status
INITIAL_LENGTH = 0.1
XPBD_FIX_POS = np.array([0.5, 0.5, 0.0])
NEWTONIAN_FIX_POS = np.array([-0.5, 0.5, 0.0])
XPBD_FREE_POS = XPBD_FIX_POS + np.array([0.0, -INITIAL_LENGTH, 0.0])
NEWTONIAN_FREE_POS = NEWTONIAN_FIX_POS + np.array([0.0, -INITIAL_LENGTH, 0.0])

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

class Particle:
    def __init__(self, position, mass=MASS, fixed=False, xpbd=False):
        self.position = position
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(3)
        self.mass = mass
        self.weight = 1.0 / mass
        self.fixed = fixed
        self.xpbd = xpbd

class Constraint:
    def __init__(self, p1, p2, compliance):
        self.p1 = p1
        self.p2 = p2
        self.compliance = compliance
        self.lambda_acc = 0.0