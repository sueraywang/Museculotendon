# SimulatorNN.py
from Physics import *

model1 = MLP(input_size=3, hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
checkpoint = torch.load('TrainedModels/Muscles/addend1.pth')
model1.load_state_dict(checkpoint['model_state_dict'])
model1.eval()

model2 = MLP(input_size=2, hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
checkpoint = torch.load('TrainedModels/Muscles/addend2.pth')
model2.load_state_dict(checkpoint['model_state_dict'])
model2.eval()

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
            vel_tensor = torch.tensor(np.dot(relative_vel, normalize(dx)) / (params.lMopt * params.vMmax), dtype=torch.float32).reshape(-1, 1)
        
        act = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
        alphaMopt = torch.tensor(params.alphaMopt, dtype=torch.float32).reshape(-1, 1)
        inputs1 = torch.cat([l_M_tendor, act, vel_tensor], dim=1).detach().requires_grad_(True)
        inputs2 = torch.cat([l_M_tendor, vel_tensor], dim=1).detach().requires_grad_(True)
        C1 = model1(inputs1)
        C2 = model2(inputs2)
        C1_values = model1(inputs1).item()
        C2_values = model2(inputs2).item()
        grad1 = torch.autograd.grad(C1, inputs1, grad_outputs=torch.ones_like(C1), create_graph=True)[0][:, 0].item()
        grad2 = torch.autograd.grad(C2, inputs2, grad_outputs=torch.ones_like(C2), create_graph=True)[0][:, 0].item()

        denominator1 = (w2 * np.dot(grad1, grad1)) + alpha 
        delta_lambda1 = -C1_values / denominator1 * math.cos(penn)
        moving_particle.position += w2 * delta_lambda1 * grad1 * n

        denominator2 = (w2 * np.dot(grad2, grad2)) + alpha 
        delta_lambda2 = -C2_values / denominator2 * math.cos(penn)
        moving_particle.position += w2 * delta_lambda2 * grad2 * n
        
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
            v_m = np.dot(relative_vel, normalize(dx))
        
        f_muscle = Muscle.muscle_force(v_m, l_M, activation, penn, params) * normalize(dx)

        # Update velocity and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * GRAVITY) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        if RENDER_MT:
            self.classic_l_M += v_m * self.sub_dt
        
        return penn