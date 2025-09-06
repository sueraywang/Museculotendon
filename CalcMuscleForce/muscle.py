import math
import numpy as np
from typing import List, Tuple
from CalcMuscleForce.utils.newton_solver import NewtonSolver
from CalcMuscleForce.curves.active_force_length import CurveActiveForceLength
from CalcMuscleForce.curves.fiber_force_length import CurveFiberForceLength
from CalcMuscleForce.curves.tendon_force_length import CurveTendonForceLength
from CalcMuscleForce.curves.force_velocity import CurveForceVelocity

class MuscleParams:
    """Container for muscle parameters"""
    
    def __init__(
        self,
        lMopt: float,
        lTslack: float,
        vMmax: float,
        alphaMopt: float,
        fMopt: float,
        beta: float,
        aMin: float,
        tauA: float,
        tauD: float,
        h: float = 0.0
    ):
        """
        Initialize muscle parameters
        
        Args:
            lMopt: Optimal muscle length
            lTslack: Tendon slack length
            vMmax: Maximum contraction velocity
            alphaMopt: Pennation angle at optimal muscle length
            fMopt: Peak isometric force
            beta: Damping coefficient
            aMin: Minimum activation
            tauA: Activation time constant
            tauD: Deactivation time constant
        """
        self.beta = beta
        self.lMopt = lMopt
        self.lTslack = lTslack
        self.vMmax = vMmax
        self.alphaMopt = alphaMopt
        self.fMopt = fMopt
        self.aMin = aMin
        self.tauA = tauA
        self.tauD = tauD
        
        # Initialize curves
        self.curve_afl = CurveActiveForceLength()
        self.curve_pfl = CurveFiberForceLength()
        self.curve_tfl = CurveTendonForceLength()
        self.curve_fv = CurveForceVelocity()
        
        # Initialize derived parameters
        self.alpha_max = math.acos(0.1)
        self.h = self.lMopt * math.sin(self.alphaMopt)
        self.l_mt = self.lTslack + self.lMopt * math.cos(self.alphaMopt)
        
        if self.alpha_max > 1e-6:
            min_pennated_fiber_length = self.h / math.sin(self.alpha_max)
        else:
            min_pennated_fiber_length = self.lMopt * 0.01
        
        min_active_fiber_length = self.curve_afl.get_min_norm_active_fiber_length() * self.lMopt
        self.l_m_min = max(min_active_fiber_length, min_pennated_fiber_length)


class Muscle:
    """Static muscle model class - all methods are static"""
    
    @staticmethod
    def calc_pennation_angle(l_m: float, params: MuscleParams) -> float:
        """
        Calculate pennation angle based on muscle length
        
        Args:
            l_m: Muscle length
            params: Muscle parameters
            
        Returns:
            Pennation angle in radians
        """
        alpha_m = 0.0
        
        if params.alphaMopt > np.finfo(float).eps:
            if l_m > params.l_m_min:
                sin_alpha = params.h / l_m
                if sin_alpha < params.alpha_max:
                    alpha_m = math.asin(sin_alpha)
                else:
                    alpha_m = params.alpha_max
            else:
                alpha_m = params.alpha_max
        
        return alpha_m
    
    @staticmethod
    def calc_pennation_angle_tilde(l_m_tilde: float, params: MuscleParams) -> float:
        """
        Calculate pennation angle based on normalized muscle length
        
        Args:
            l_m_tilde: Normalized muscle length (l_m / lMopt)
            params: Muscle parameters
            
        Returns:
            Pennation angle in radians
        """
        alpha_m = 0.0
        
        if params.alphaMopt > np.finfo(float).eps:
            h_tilde = params.h / params.lMopt
            if l_m_tilde > params.l_m_min / params.lMopt:
                sin_alpha = h_tilde / l_m_tilde
                if sin_alpha < params.alpha_max:
                    alpha_m = math.asin(sin_alpha)
                else:
                    alpha_m = params.alpha_max
            else:
                alpha_m = params.alpha_max
        
        return alpha_m
    
    @staticmethod
    def force_balance(
        v_m_tilde: float,
        act: float,
        afl: float,
        pfl: float,
        tfl: float,
        curve_fv: CurveForceVelocity,
        cos_alpha_m: float,
        beta: float
    ) -> Tuple[float, float]:
        """
        Force balance function for Newton solver
        
        Args:
            v_m_tilde: Normalized muscle velocity
            act: Activation level
            afl: Active force-length multiplier
            pfl: Passive force-length multiplier
            tfl: Tendon force-length multiplier
            curve_fv: Force-velocity curve
            cos_alpha_m: Cosine of pennation angle
            beta: Damping coefficient
            
        Returns:
            Tuple of (f, J) where f is the force balance equation and J is its derivative
        """
        deriv = curve_fv.calc_val_deriv(v_m_tilde)
        fv = deriv[0]
        dfv = deriv[1]
        
        f_m = act * afl * fv + pfl + beta * v_m_tilde
        f = f_m * cos_alpha_m - tfl
        J = (act * afl * dfv + beta) * cos_alpha_m
        
        return f, J
    
    @staticmethod
    def compute_vel_tilde(
        l_m_tilde: float,
        l_t_tilde: float,
        act: float,
        alpha_m: float,
        params: MuscleParams
    ) -> float:
        """
        Compute normalized muscle velocity
        
        Args:
            l_m_tilde: Normalized muscle length
            l_t_tilde: Normalized tendon length
            act: Activation level
            alpha_m: Pennation angle
            params: Muscle parameters
            
        Returns:
            Normalized muscle velocity
        """
        cos_alpha_m = math.cos(alpha_m)
        afl = params.curve_afl.calc_value(l_m_tilde)
        pfl = params.curve_pfl.calc_value(l_m_tilde)
        tfl = params.curve_tfl.calc_value(l_t_tilde)
        v_m_tilde_init = 0.0
        
        # Define the objective function for Newton solver
        def objective(v_m_tilde):
            return Muscle.force_balance(
                v_m_tilde, act, afl, pfl, tfl, params.curve_fv, cos_alpha_m, params.beta
            )
        
        # Use Newton solver to find the muscle fiber velocity
        v_m_tilde = NewtonSolver.solve(objective, v_m_tilde_init)
        
        return v_m_tilde
    
    @staticmethod
    def compute_vel(
        l_m: float, 
        l_mt: float, 
        act: float, 
        alpha_m: float,
        params: MuscleParams
    ) -> float:
        """
        Compute muscle velocity
        
        Args:
            l_m: Muscle length
            l_mt: Musculotendon length
            act: Activation level
            alpha_m: Pennation angle
            params: Muscle parameters
            
        Returns:
            Muscle velocity
        """
        l_m_tilde = l_m / params.lMopt
        l_t = l_mt - l_m * math.cos(alpha_m)
        l_t_tilde = l_t / params.lTslack
        
        v_m_tilde = Muscle.compute_vel_tilde(l_m_tilde, l_t_tilde, act, alpha_m, params)
        v_m = v_m_tilde * params.lMopt * params.vMmax
        
        return v_m
    
    @staticmethod
    def muscle_force(
        v_m: float, 
        l_m: float, 
        act: float, 
        alpha_m: float,
        params: MuscleParams
    ) -> float:
        """
        Calculate muscle force
        
        Args:
            v_m: Muscle velocity
            l_m: Muscle length
            act: Activation level
            alpha_m: Pennation angle
            params: Muscle parameters
            
        Returns:
            Muscle force
        """
        l_m_tilde = l_m / params.lMopt
        v_m_tilde = v_m / (params.lMopt * params.vMmax)
        
        afl = params.curve_afl.calc_value(l_m_tilde)
        pfl = params.curve_pfl.calc_value(l_m_tilde)
        deriv = params.curve_fv.calc_val_deriv(v_m_tilde)
        fv = deriv[0]
        
        f_m = act * afl * fv + pfl + params.beta * v_m_tilde
        f = f_m * math.cos(alpha_m) * params.fMopt
        
        return f
    
    @staticmethod
    def muscle_force_vectorized(
        v_m: List[float],
        l_m: List[float],
        act: List[float],
        alpha_m: List[float],
        params: MuscleParams
    ) -> List[float]:
        """
        Vectorized muscle force calculation for performance
        
        Args:
            v_m: List of muscle velocities
            l_m: List of muscle lengths
            act: List of activation levels
            alpha_m: List of pennation angles
            params: Muscle parameters
            
        Returns:
            List of muscle forces
        """
        n = len(v_m)
        forces = [0.0] * n
        
        for i in range(n):
            forces[i] = Muscle.muscle_force(v_m[i], l_m[i], act[i], alpha_m[i], params)
        
        return forces