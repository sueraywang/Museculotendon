# SimulatorNN.py
from Physics import *

model = MLP(input_size=model_params['input_size'], hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
checkpoint = torch.load(os.path.join('TrainedModels/Muscles', model_params['model_name']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class Simulator:
    def __init__(self, l_M):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.penn = 0.0
        self.classic_l_M = l_M
        self.xpbd_l_M = l_M

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
        relative_vel = fixed_particle.velocity - moving_particle.velocity

        if RENDER_MT:
            l_MT = np.linalg.norm(dx)
            penn = Muscle.calc_pennation_angle(self.xpbd_l_M, params)
            l_T = l_MT - self.xpbd_l_M * math.cos(penn)

            # Ensure valid tendon length
            if l_T < params.lTslack:
                l_T = params.lTslack
                self.xpbd_l_M = (l_MT - l_T) / math.cos(penn)
                penn = Muscle.calc_pennation_angle(self.xpbd_l_M, params)

            # Calculate muscle velocity using your class
            v_m = Muscle.compute_vel(self.xpbd_l_M, l_MT, activation, penn, params)
        else:
            penn = Muscle.calc_pennation_angle(np.linalg.norm(dx), params)

        w2 = moving_particle.weight
        alpha = MUSCLE_COMPLIANCE / (self.sub_dt * self.sub_dt)
        beta = -params.beta * self.sub_dt**2 / params.lMopt / params.vMmax * params.fMopt * math.cos(penn)
        n = dx / np.linalg.norm(dx)
        #gamma = alpha * beta / self.sub_dt
        
        if RENDER_MT:
            l_M_tendor = torch.tensor(self.xpbd_l_M/params.lMopt, dtype=torch.float32).reshape(-1, 1)  # Physical units
            vel_tensor = torch.tensor(v_m / (params.lMopt * params.vMmax), dtype=torch.float32).reshape(-1, 1)
        else:
            l_M_tendor = torch.tensor(np.linalg.norm(dx)/params.lMopt, dtype=torch.float32).reshape(-1, 1)
            vel_tensor = torch.tensor(np.dot(relative_vel, normalized(dx)) / (params.lMopt * params.vMmax), dtype=torch.float32).reshape(-1, 1)
        
        act = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
        alphaMopt = torch.tensor(params.alphaMopt, dtype=torch.float32).reshape(-1, 1)
        inputs = torch.cat([l_M_tendor, act, vel_tensor], dim=1).detach().requires_grad_(True)
        C = model(inputs)
        C_values = model(inputs).item()
        grad = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 0].item()

        denominator = (w2 * np.dot(grad, grad)) + alpha
        #denom1 = (1 + gamma) * (w1 * np.dot(grad, grad) + w2 * np.dot(grad, grad)) + alpha    
        delta_lambda = -C_values / denominator * math.cos(penn)
        moving_particle.position += w2 * delta_lambda * grad * n
        
        #"""
        # Update damping
        grad = normalized(dx)# relative_vel / np.linalg.norm(relative_vel)
        delta_lambda_damping = -(beta/self.sub_dt * np.dot(grad, relative_motion)) / (beta/self.sub_dt * w2 + 1)
        moving_particle.position += w2 * delta_lambda_damping * grad
        #"""
        
        moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt

        if RENDER_MT:
            self.xpbd_l_M += v_m * self.sub_dt
        
        return penn

    def classic_substep(self, activation):
        fixed_particle = self.particles[2]
        moving_particle = self.particles[3]
        dx = fixed_particle.position - moving_particle.position
        if RENDER_MT:
            l_MT = np.linalg.norm(dx)

            # Calculate pennation angle and tendon length
            penn = Muscle.calc_pennation_angle(self.classic_l_M, params)
            l_T = l_MT - self.classic_l_M * math.cos(penn)

            # Ensure valid tendon length
            if l_T < params.lTslack:
                l_T = params.lTslack
                self.classic_l_M = (l_MT - l_T) / math.cos(alpha_m)
                alpha_m = Muscle.calc_pennation_angle(self.classic_l_M, params)
            
            # Calculate muscle velocity using your class
            v_m = Muscle.compute_vel(self.classic_l_M, l_MT, activation, penn, params)
            l_M = self.classic_l_M
        else:
            l_M = np.linalg.norm(dx)
            penn = np.arcsin(params.h/l_M)
            relative_vel = fixed_particle.velocity - moving_particle.velocity
            v_m = np.dot(relative_vel, normalized(dx))
        
        f_muscle = Muscle.muscle_force(v_m, l_M, activation, penn, params) * normalized(dx)

        # Update velocity and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * GRAVITY) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        if RENDER_MT:
            self.classic_l_M += v_m * self.sub_dt
        
        return penn