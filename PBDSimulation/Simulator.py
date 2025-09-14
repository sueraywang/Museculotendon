# SimulatorNN.py
from Physics import *

class Simulator:
    def __init__(self, l_M, penn_model, damping_model, nn_model):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.alpha = MUSCLE_COMPLIANCE / (self.sub_dt * self.sub_dt)
        self.penn = 0.0
        self.classic_l_M = l_M
        self.xpbd_l_M = l_M
        self.damping_model = damping_model #denotes whether the NN model include damping
        self.penn_model = penn_model #denotes whether the NN model include pennation angle
        self.nn_model = nn_model

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

        MT_vec = fixed_particle.position - moving_particle.position
        MT_dir = normalize(MT_vec)
        w2 = moving_particle.weight

        if RENDER_MT:
            muscle_vel_scalar = Muscle.compute_vel(self.xpbd_l_M, np.linalg.norm(MT_vec), activation, self.penn, params)
            self.xpbd_l_M += muscle_vel_scalar * self.sub_dt
            l_M = self.xpbd_l_M
        else:
            l_M = np.linalg.norm(MT_vec)
            muscle_vel_scalar = np.dot(fixed_particle.velocity - moving_particle.velocity, MT_dir)
        
        penn = Muscle.calc_pennation_angle(l_M, params)
        act_tendor = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
        l_M_tendor = torch.tensor(l_M/params.lMopt, dtype=torch.float32).reshape(-1, 1)
        vel_tensor = torch.tensor(muscle_vel_scalar / (params.lMopt * params.vMmax), dtype=torch.float32).reshape(-1, 1)

        if self.penn_model:
            alphaMopt_tendor = torch.tensor(params.alphaMopt, dtype=torch.float32).reshape(-1, 1)
            inputs = torch.cat([l_M_tendor, act_tendor, vel_tensor, alphaMopt_tendor], dim=1).detach().requires_grad_(True)
        else:
            inputs = torch.cat([l_M_tendor, act_tendor, vel_tensor], dim=1).detach().requires_grad_(True)
        
        C = self.nn_model(inputs)
        C_values = self.nn_model(inputs).item()
        grad = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 0].item()
        denominator = (w2 * np.dot(grad, grad)) + self.alpha 

        if self.penn_model:
            delta_lambda = -C_values / denominator
        else:
            delta_lambda = -C_values / denominator * math.cos(penn)

        moving_particle.position += w2 * delta_lambda * grad * MT_dir
        
        if not self.damping_model:
            MT_vec_prev = fixed_particle.prev_position-moving_particle.prev_position
            delta_MT_vec = MT_vec - MT_vec_prev
            beta = -params.beta * self.sub_dt**2 / (params.lMopt * params.vMmax) * params.fMopt * math.cos(penn)
            delta_lambda_damping = -(beta/self.sub_dt * np.dot(MT_dir, delta_MT_vec)) / (beta/self.sub_dt * w2 + 1)
            moving_particle.position += w2 * delta_lambda_damping * MT_dir
        
        moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
        
        return penn

    def classic_substep(self, activation):
        fixed_particle = self.particles[2]
        moving_particle = self.particles[3]
        MT_vec = fixed_particle.position - moving_particle.position
        MT_dir = normalize(MT_vec)
            
        if RENDER_MT:
            muscle_vel_scalar = Muscle.compute_vel(self.classic_l_M, np.linalg.norm(MT_vec), activation, self.penn, params)
            self.classic_l_M += muscle_vel_scalar * self.sub_dt
            l_M = self.classic_l_M

        else:
            l_M = np.linalg.norm(MT_vec)
            muscle_vel_scalar = np.dot(fixed_particle.velocity - moving_particle.velocity, MT_dir)
        
        penn = Muscle.calc_pennation_angle(l_M, params)
        f_muscle = Muscle.muscle_force(muscle_vel_scalar, l_M, activation, penn, params) * MT_dir

        # Update velocity and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * GRAVITY) * moving_particle.weight) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        
        return penn