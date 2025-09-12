# SimulatorNN.py
from Physics import *

if RENDER_MT:
    model = MLP(input_size=3, hidden_size=256, 
            output_size=1, num_layers=5, 
            activation='tanh')
    checkpoint = torch.load(os.path.join('TrainedModels/Muscles', "lm_lmt_model.pth"))
else: 
    model = MLP(input_size=model_params['input_size'], hidden_size=model_params['num_width'], 
                output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
                activation=model_params['activation_func'])
    checkpoint = torch.load(os.path.join('TrainedModels/Muscles', model_params['model_name']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class Simulator:
    def __init__(self):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
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
        relative_vel = fixed_particle.velocity - moving_particle.velocity
        penn = np.arcsin(params.h/np.linalg.norm(dx))
        n = dx / np.linalg.norm(dx)

        w2 = moving_particle.weight
        alpha = MUSCLE_COMPLIANCE / (self.sub_dt * self.sub_dt)
        beta = -params.beta * self.sub_dt**2 / params.lMopt / params.vMmax * params.fMopt * math.cos(penn)
        #gamma = alpha * beta / self.sub_dt
        
        if (np.linalg.norm(dx)/params.lMopt) < 0 or (np.linalg.norm(dx)/params.lMopt) > 2:
            print(f"lMtilde exceeds the bound. It's {(np.linalg.norm(dx)/params.lMopt)}") 
        
        dx_tensor = torch.tensor(np.linalg.norm(dx)/params.lMopt, dtype=torch.float32).reshape(-1, 1)
        vel_tensor = torch.tensor(np.dot(relative_vel, normalized(dx)) / (params.lMopt * params.vMmax), dtype=torch.float32).reshape(-1, 1)
        activation = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
        alphaMopt = torch.tensor(params.alphaMopt, dtype=torch.float32).reshape(-1, 1)
        inputs = torch.cat([dx_tensor, activation, vel_tensor], dim=1).detach().requires_grad_(True)
        C = model(inputs)
        C_values = model(inputs).item()
        grad = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 0].item() 
        denominator = (w2 * np.dot(grad, grad)) + alpha
        #denom1 = (1 + gamma) * (w1 * np.dot(grad, grad) + w2 * np.dot(grad, grad)) + alpha
        
        delta_lambda = -C_values / denominator * math.cos(penn)
        moving_particle.position += w2 * delta_lambda * grad * n
        
        """
        loss = muscle_force_full(np.linalg.norm(dx)/params.lMopt, activation, np.dot(relative_vel, normalized(dx)) / (params.lMopt * params.vMmax), penn) + C_values * grad
        print(loss)
        """
        
        #"""
        # Update damping
        grad = normalized(dx)# relative_vel / np.linalg.norm(relative_vel)
        delta_lambda_damping = -(beta/self.sub_dt * np.dot(grad, relative_motion)) / (beta/self.sub_dt * w2 + 1)
        moving_particle.position += w2 * delta_lambda_damping * grad
        #"""
        
        moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
        
        return penn

    def classic_substep(self, activation):
        fixed_particle = self.particles[2]
        moving_particle = self.particles[3]
        dx = fixed_particle.position - moving_particle.position
        l_m = np.linalg.norm(dx)
        penn = np.arcsin(params.h/l_m)
        relative_vel = fixed_particle.velocity - moving_particle.velocity
        v_m = np.dot(relative_vel, normalized(dx))
        
        f_muscle = Muscle.muscle_force(v_m, l_m, activation, penn, params) * normalized(dx)

        # Update velocity and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * GRAVITY) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        
        return penn