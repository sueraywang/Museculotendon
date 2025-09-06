import time
import numpy as np
from MuscleNN3D import Simulator
from Renderer import Renderer
from Plotter import SimulationPlotter
from Physics import *
from scipy.optimize import brentq

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
            alpha_m = Muscle.calc_pennation_angle(params.lMopt)
            l_T = l_MT - params.lMopt * math.cos(alpha_m)
            if l_T < params.lTslack:
                l_T = params.lTslack
                return (l_MT - l_T) / math.cos(alpha_m)
            return params.lMopt

def main():
    # Initialize the simulator
    simulator = Simulator()
    
    # Create the renderer with the new ModernOpenGLRenderer
    renderer = Renderer(simulator, cam_distance=3.0)
    
    # Create a plotter for data visualization
    plotter = SimulationPlotter()
    
    t = 0.0
    
    # Main simulation loop
    while t <= 10.0 and not renderer.should_close():
        # Process input
        renderer.process_input()
        
        # Calculate activation and pennation for plotting
        activation = np.sin(plotter.plot_time * 20) / 2.0 + 0.5
        
        # Render the scene
        renderer.render()
        
        # Update the plot
        plotter.update(simulator, DT, activation, simulator.penn)
        
        # Update the simulation
        simulator.step(activation)
        
        # Maintain consistent frame rate
        time.sleep(DT)
        t += DT

    # Export the simulation data
    #plotter.export_data("precomputed_model_100steps.csv")
    
    # Clean up resources
    renderer.cleanup()

main()