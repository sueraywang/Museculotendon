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
from scipy.optimize import brentq

# Create muscle parameters
params = MuscleParams(
    lMopt=0.1,      # optimal muscle length
    lTslack=0.2,    # tendon slack length
    vMmax=10.0,     # maximum contraction velocity
    alphaMopt=0.0,  # pennation angle at optimal muscle length
    fMopt=2.0,      # peak isometric force, originally = 100
    beta=0.1,       # damping
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
RENDER_MT = True
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
    
def find_initial_fiber_equilibrium(l_MT, activation):
    """
    Find the initial fiber length that satisfies force equilibrium
    """
    
    def equilibrium_error(l_m):
        """Calculate force balance error - FIXED VERSION"""
        try:
            # Calculate pennation angle
            alpha_m = Muscle.calc_pennation_angle(l_m)
            
            # Calculate tendon length from geometry
            l_T = l_MT - l_m * math.cos(alpha_m)
            
            # Check if tendon length is valid
            if l_T <= params.lTslack:
                return 1e6
            
            # Normalized lengths
            l_m_tilde = l_m / params.lMopt
            l_T_tilde = l_T / params.lTslack
            
            # THE KEY FIX: Calculate forces assuming ZERO VELOCITY (fv = 1.0)
            # Don't use compute_vel_Tilde here - it creates circular dependency!
            afl = params.curve_afl.calc_value(l_m_tilde)
            pfl = params.curve_pfl.calc_value(l_m_tilde)
            tfl = params.curve_tfl.calc_value(l_T_tilde)
            
            # Fiber force assuming fv = 1.0 (zero velocity at equilibrium)
            f_fiber_normalized = activation * afl + pfl  # No fv term, no damping
            
            # Fiber force along tendon direction
            f_fiber_along_tendon = f_fiber_normalized * math.cos(alpha_m)
            
            # Force balance error: fiber force should equal tendon force
            error = abs(f_fiber_along_tendon - tfl)
            
            return error
            
        except Exception as e:
            return 1e6
    
    # IMPROVED: Better search bounds for high activation
    l_m_min = params.lMopt * 0.5
    l_m_max = params.lMopt * 1.5
    
    # For high activation, adjust bounds because muscle will be shorter
    if activation > 0.5:
        l_m_min = params.lMopt * 0.3  # Allow shorter fibers
        l_m_max = params.lMopt * 1.2  # Don't need to go as long
    
    # IMPROVED: Use bracketing search (more robust than minimize_scalar)
    n_search = 100
    l_m_test = np.linspace(l_m_min, l_m_max, n_search)
    
    # Find where the error function changes sign
    for i in range(len(l_m_test) - 1):
        # Calculate residual (not absolute error) to find sign changes
        def residual(l_m):
            try:
                alpha_m = Muscle.calc_pennation_angle(l_m)
                l_T = l_MT - l_m * math.cos(alpha_m)
                if l_T <= params.lTslack:
                    return 1e6
                
                l_m_tilde = l_m / params.lMopt
                l_T_tilde = l_T / params.lTslack
                
                afl = params.curve_afl.calc_value(l_m_tilde)
                pfl = params.curve_pfl.calc_value(l_m_tilde)
                tfl = params.curve_tfl.calc_value(l_T_tilde)
                
                f_fiber_normalized = activation * afl + pfl
                f_fiber_along_tendon = f_fiber_normalized * math.cos(alpha_m)
                
                return f_fiber_along_tendon - tfl  # Note: not absolute value
                
            except Exception:
                return 1e6
        
        r1 = residual(l_m_test[i])
        r2 = residual(l_m_test[i + 1])
        
        # Found a sign change - there's a root here!
        if abs(r1) < 1e6 and abs(r2) < 1e6 and r1 * r2 < 0:
            try:
                solution = brentq(residual, l_m_test[i], l_m_test[i + 1], xtol=1e-12)
                if equilibrium_error(solution) < 1e-8:
                    return solution
            except:
                continue
    
    # Fallback: use minimize_scalar on the absolute error
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(
        equilibrium_error,
        bounds=(l_m_min, l_m_max),
        method='bounded'
    )
    
    if result.success and result.fun < 1e-8:
        return result.x
    else:
        # Final fallback - but make it smarter for high activation
        print(f"Warning: Force equilibrium not found, using smart fallback")
        if activation > 0.5:
            # High activation = high muscle force = more tendon stretch = shorter fiber
            l_T_stretched = params.lTslack * (1.0 + 0.02 * activation)  # 2% stretch per unit activation
            return (l_MT - l_T_stretched) / math.cos(params.alphaMopt)
        else:
            # Low activation = use your original geometric solution
            alpha_m = Muscle.calc_pennation_angle(params.lMopt, params)
            l_T = l_MT - params.lMopt * math.cos(alpha_m)
            if l_T < params.lTslack:
                l_T = params.lTslack
                return (l_MT - l_T) / math.cos(alpha_m)
            return params.lMopt