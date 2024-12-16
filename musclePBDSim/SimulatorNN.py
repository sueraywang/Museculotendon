# SimulatorNN.py
from Physics import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize model with the same architecture as during training
model = MLP(hidden_size=128)
checkpoint = torch.load('musclePBDSim/dampedLinearSpringBestModel.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class Simulator:
    def __init__(self):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.gravity = GRAVITY

        self.particles = [
            Particle(XPBD_FIX_POS.copy(), fixed=True, xpbd=True),   
            Particle(XPBD_FREE_POS.copy(), xpbd=True),
            Particle(CLASSIC_FIX_POS.copy(), fixed=True),
            Particle(CLASSIC_FREE_POS.copy())
        ]

        self.constraints = [
            Constraint(self.particles[0], self.particles[1], COMPLIANCE)
        ]

    def step(self):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0

        for _ in range(self.num_substeps):
            self.xpbd_substep()
            self.classic_substep()

    def xpbd_substep(self):
        for particle in self.particles:
            if particle.xpbd:
                particle.prev_position = particle.position.copy()
                particle.velocity += self.gravity * self.sub_dt
                particle.position += particle.velocity * self.sub_dt

        for constraint in self.constraints:
            x1, x2 = constraint.p1.position, constraint.p2.position
            dx = x1 - x2
            n = dx / np.linalg.norm(dx)
            
            x1_prev, x2_prev = constraint.p1.prev_position, constraint.p2.prev_position
            dx_prev = x2_prev - x1_prev
            l_prev = np.linalg.norm(dx_prev)
            grad_dx = np.linalg.norm(dx) - l_prev

            dx_tensor = torch.tensor(np.linalg.norm(dx) - REST_LENGTH, dtype=torch.float32).reshape(-1, 1)
            dv_tensor = torch.tensor(0.00, dtype=torch.float32).reshape(-1, 1)
            dx_prev_tensor = torch.tensor(np.linalg.norm(dx_prev) - REST_LENGTH, dtype=torch.float32).reshape(-1, 1)
            inputs = torch.cat([dx_tensor, dv_tensor], dim=1).detach().requires_grad_(True)
            inputs_prev = torch.cat([dx_prev_tensor, dv_tensor], dim=1).detach().requires_grad_(True)
            C_values = model(inputs)
            C_prev = model(inputs_prev)
            grad1 = n * torch.autograd.grad(C_values, inputs, grad_outputs=torch.ones_like(C_values), create_graph=True)[0][:, 0].item() 
            grad2 = -grad1
            grad_prev = torch.autograd.grad(C_prev, inputs_prev, grad_outputs=torch.ones_like(C_prev), create_graph=True)[0][:, 0].item()

            w1, w2 = constraint.p1.weight, constraint.p2.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            beta = DAMPING_CONSTANT * self.sub_dt**2
            gamma = (alpha * beta)/self.sub_dt 
            denominator = (1 + gamma) * (w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2)) + alpha

            if denominator == 0:
                continue

            delta_lambda = -(C_values.item() + alpha * constraint.lambda_acc + gamma * grad_prev) / denominator
            constraint.p1.position += w1 * delta_lambda * grad1
            constraint.p2.position += w2 * delta_lambda * grad2  

            d = 0
            for particle in self.particles:
                if particle.fixed and particle.xpbd:
                    d = particle.position - particle.prev_position
            for particle in self.particles:
                if particle.xpbd:
                    particle.velocity = (particle.position - particle.prev_position) / self.sub_dt
                    particle.position -= d

    def classic_substep(self):
        moving_particle = self.particles[3]
        fixed_particle = self.particles[2]
        spring_force = compute_spring_force(moving_particle.position, fixed_particle.position, 
                                         SPRING_CONSTANT, REST_LENGTH) - DAMPING_CONSTANT * (moving_particle.velocity - fixed_particle.velocity)

        moving_particle.velocity += ((spring_force + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        fixed_particle.velocity += ((-spring_force + fixed_particle.mass * self.gravity) / fixed_particle.mass) * self.sub_dt
        d = fixed_particle.velocity * self.sub_dt
        moving_particle.position -= d