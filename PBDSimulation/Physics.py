# Physics.py
import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from CalcMuscleForce.muscle import *
import torch
import torch.nn as nn
import torch.nn.functional as F 

# Create muscle parameters
params = MuscleParams(
    lMopt=0.1,      # optimal muscle length
    lTslack=0.2,    # tendon slack length
    vMmax=10.0,     # maximum contraction velocity
    alphaMopt=0.2,  # pennation angle at optimal muscle length
    fMopt=2.0,      # peak isometric force, originally = 100
    beta=0.9,       # damping
    tauA = 0.01,    # activation constant
    tauD = 0.4,     # deactivation constant
    aMin = 0.01
)
params.h = params.lMopt * np.sin(params.alphaMopt)

# Model Size
model_params = {
    'num_layer' : 3,
    'num_width' : 64,
    'activation_func' : 'tanh',
    'input_size' : 3,
    'output_size' : 1,
    'model_name' : 'len_act_vel_model.pth' 
}

# Physical constants
MASS = 0.1  # Mass of particle (kg)
GRAVITY = np.array([0, -9.8, 0])

# XPBD constants
DT = 0.01
SUB_STEPS = 100
MUSCLE_COMPLIANCE = 1/params.fMopt

# Muscle' initial status
RENDER_MT = False
if RENDER_MT:
    INITIAL_LENGTH = 0.3
else:
    INITIAL_LENGTH = 0.1
XPBD_FREE_POS = np.array([0.5, 1.0, 0.0])
CLASSIC_FREE_POS = np.array([-0.5, 1.0, 0.0])
XPBD_FIX_POS = XPBD_FREE_POS + np.array([0.0, INITIAL_LENGTH, 0.0])
CLASSIC_FIX_POS = CLASSIC_FREE_POS + np.array([0.0, INITIAL_LENGTH, 0.0])
FREE_PARTICAL_INITIAL_VEL = np.array([0.1, 0.1, 0.1])

# Helper functions
def normalized(vec):
    return vec / np.linalg.norm(vec)

class Particle:
    def __init__(self, position, mass=MASS, fixed=False, xpbd=False):
        self.position = position
        self.prev_position = self.position.copy()
        self.fixed = fixed
        if self.fixed is True:
            self.velocity = np.zeros(3)
        else:
            self.velocity = FREE_PARTICAL_INITIAL_VEL
        self.prev_velocity = self.velocity.copy()
        self.mass = mass
        self.weight = 1.0 / mass if mass != 0.0 else 0.0
        self.fixed = fixed
        self.xpbd = xpbd
        
class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=1, num_layers=3, activation='relu'):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Create layers dynamically
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for i in range(num_layers - 2):  # -2 because we have input and output layers
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
        # Set activation function
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.elu  # Default
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # All layers except the last one get activation
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x