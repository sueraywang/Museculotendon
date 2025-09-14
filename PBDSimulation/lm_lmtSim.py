import time
import numpy as np
from Renderer import Renderer
from Plotter import SimulationPlotter
from Physics import *
from scipy.optimize import brentq

model = MLP(input_size=3, hidden_size=256, 
            output_size=1, num_layers=5, 
            activation='tanh')
checkpoint = torch.load(os.path.join('TrainedModels/Muscles', "lm_lmt_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def find_initial_fiber_equilibrium(l_MT, activation):
    """
    FIXED VERSION: Find the initial fiber length that satisfies force equilibrium
    
    The key fix: Don't use compute_vel_Tilde in equilibrium calculation!
    Instead, assume zero velocity (fv = 1.0) for initial equilibrium.
    """
    
    def equilibrium_error(l_m):
        """Calculate force balance error - FIXED VERSION"""
        try:
            # Calculate pennation angle
            alpha_m = Muscle.calc_pennation_angle(l_m, params)
            
            # Calculate tendon length from geometry
            l_T = l_MT - l_m * math.cos(alpha_m)
            
            # Check if tendon length is valid
            if l_T <= params.lTslack:
                return 1e6
            
            # normalize lengths
            l_m_tilde = l_m / params.lMopt
            l_T_tilde = l_T / params.lTslack
            
            # THE KEY FIX: Calculate forces assuming ZERO VELOCITY (fv = 1.0)
            # Don't use compute_vel_Tilde here - it creates circular dependency!
            afl = params.curve_afl.calc_value(l_m_tilde)
            pfl = params.curve_pfl.calc_value(l_m_tilde)
            tfl = params.curve_tfl.calc_value(l_T_tilde)
            
            # Fiber force assuming fv = 1.0 (zero velocity at equilibrium)
            f_fiber_normalize = activation * afl + pfl  # No fv term, no damping
            
            # Fiber force along tendon direction
            f_fiber_along_tendon = f_fiber_normalize * math.cos(alpha_m)
            
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
                alpha_m = Muscle.calc_pennation_angle(l_m, params)
                l_T = l_MT - l_m * math.cos(alpha_m)
                if l_T <= params.lTslack:
                    return 1e6
                
                l_m_tilde = l_m / params.lMopt
                l_T_tilde = l_T / params.lTslack
                
                afl = params.curve_afl.calc_value(l_m_tilde)
                pfl = params.curve_pfl.calc_value(l_m_tilde)
                tfl = params.curve_tfl.calc_value(l_T_tilde)
                
                f_fiber_normalize = activation * afl + pfl
                f_fiber_along_tendon = f_fiber_normalize * math.cos(alpha_m)
                
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
        
class Simulator:
    def __init__(self, l_M):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.classic_l_M = l_M
        self.xpbd_l_M = l_M
        self.penn = 0.0

        self.particles = [
            Particle(XPBD_FIX_POS.copy(), fixed=True, xpbd=True),   
            Particle(XPBD_FREE_POS.copy(), xpbd=True),
            Particle(CLASSIC_FIX_POS.copy(), fixed=True),
            Particle(CLASSIC_FREE_POS.copy())
        ]
        
    def step(self, activation):
        for _ in range(self.num_substeps):
            penn1 = self.xpbd_substep(activation)
            penn2 = self.classic_substep(activation)
            if penn1 - penn2 < 1e-3:
                self.penn = penn1   
            
    def xpbd_substep(self, activation):
        fixed_particle = self.particles[0]
        moving_particle = self.particles[1]
        moving_particle.prev_position = moving_particle.position.copy()
        moving_particle.velocity += GRAVITY * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt

        dx = fixed_particle.position - moving_particle.position
        dx_prev = fixed_particle.prev_position-moving_particle.prev_position
        relative_motion = dx - dx_prev
        l_MT = np.linalg.norm(dx)
    
        # Calculate pennation angle and tendon length
        alpha_m = Muscle.calc_pennation_angle(self.xpbd_l_M, params)
        l_T = l_MT - self.xpbd_l_M * math.cos(alpha_m)
        
        # Ensure valid tendon length
        if l_T < params.lTslack:
            l_T = params.lTslack
            self.xpbd_l_M = (l_MT - l_T) / math.cos(alpha_m)
            alpha_m = Muscle.calc_pennation_angle(self.xpbd_l_M, params)
        
        # Calculate muscle velocity using your class
        v_m = Muscle.compute_vel(self.xpbd_l_M, l_MT, activation, alpha_m, params)
        
        n = dx / np.linalg.norm(dx)
        w2 = moving_particle.weight
        alpha = MUSCLE_COMPLIANCE / (self.sub_dt * self.sub_dt)
        beta = -params.beta * self.sub_dt**2 / params.lMopt / params.vMmax * params.fMopt * math.cos(alpha_m)
        
        if (self.xpbd_l_M/params.lMopt) < 0 or (self.xpbd_l_M/params.lMopt) > 2:
            print(f"lMtilde exceeds the bound. It's {(np.linalg.norm(dx)/params.lMopt)}") 
    
        l_MT_tensor = torch.tensor(l_MT, dtype=torch.float32).reshape(-1, 1)
        l_M_tensor = torch.tensor(self.xpbd_l_M, dtype=torch.float32).reshape(-1, 1)  # Physical units
        act = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
        inputs = torch.cat([l_MT_tensor, l_M_tensor, act], dim=1).detach().requires_grad_(True)

        # Get constraint and gradient
        C = model(inputs)
        C_values = C.item()

        # Get dC/dl_M (gradient w.r.t. physical l_M)
        dC_dl_M = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 1].item()

        grad = dC_dl_M * params.lMopt 
        denominator = (w2 * np.dot(grad, grad)) + alpha
        
        delta_lambda = -C_values / denominator# * math.cos(alpha_m)
        moving_particle.position += w2 * delta_lambda * grad * n
        
        """
        # Update damping
        grad = normalize(dx)# relative_vel / np.linalg.norm(relative_vel)
        delta_lambda_damping = -(beta/self.sub_dt * np.dot(grad, relative_motion)) / (beta/self.sub_dt * w2 + 1)
        moving_particle.position += w2 * delta_lambda_damping * grad
        #"""
        
        moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
        self.xpbd_l_M += v_m * self.sub_dt
            
        return alpha_m
    
    def classic_substep(self, activation):
        fixed_particle = self.particles[2]
        moving_particle = self.particles[3]
        dx = fixed_particle.position - moving_particle.position
        l_MT = np.linalg.norm(dx)
        
        # Calculate pennation angle and tendon length
        alpha_m = Muscle.calc_pennation_angle(self.classic_l_M, params)
        l_T = l_MT - self.classic_l_M * math.cos(alpha_m)
        
        # Ensure valid tendon length
        if l_T < params.lTslack:
            l_T = params.lTslack
            self.classic_l_M = (l_MT - l_T) / math.cos(alpha_m)
            alpha_m = Muscle.calc_pennation_angle(self.classic_l_M, params)
        
        # Calculate muscle velocity using your class
        v_m = Muscle.compute_vel(self.classic_l_M, l_MT, activation, alpha_m, params)
        
        # Calculate muscle force
        f_muscle = Muscle.muscle_force(v_m, self.classic_l_M, activation, alpha_m, params) * normalize(dx)
        
        # Update velocity, l_M, and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * GRAVITY) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        self.classic_l_M += v_m * self.sub_dt
        
        return alpha_m
    
def main():
    
    l_M = find_initial_fiber_equilibrium(INITIAL_LENGTH, 0.5)
    
    # Initialize the simulator
    simulator = Simulator(l_M)
    
    # Create the renderer with the new ModernOpenGLRenderer
    renderer = Renderer(simulator, cam_distance=3.0, MT=True)
    
    # Create a plotter for data visualization
    plotter = SimulationPlotter(MT=True)
    
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
    #plotter.export_data("lm_lmt_model_100steps.csv")
    
    # Clean up resources
    renderer.cleanup()

main()