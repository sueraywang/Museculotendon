# SimulatorNN.py
from Physics import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
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
checkpoint = torch.load('musclePBDSim/springForceBestModel.pth')
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
            Particle(NEWTONIAN_FIX_POS.copy(), fixed=True),
            Particle(NEWTONIAN_FREE_POS.copy())
        ]

        self.constraints = [
            Constraint(self.particles[0], self.particles[1], COMPLIANCE)
        ]

    def step(self):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0

        for _ in range(self.num_substeps):
            self.xpbd_substep()
            self.newtonian_substep()

    def xpbd_substep(self):
        for particle in self.particles:
            if particle.xpbd:
                particle.prev_position = particle.position.copy()
                particle.velocity += self.gravity * self.sub_dt
                particle.position += particle.velocity * self.sub_dt

        for constraint in self.constraints:
            x1, x2 = constraint.p1.position, constraint.p2.position
            diff = x1 - x2
            current_length = np.linalg.norm(diff)

            if current_length < 1e-7:
                continue

            displacement = current_length - REST_LENGTH
            n = diff / current_length

            dx_tensor = dx_tensor = torch.tensor([[displacement]], dtype=torch.float32)  # Convert to 2D tensor
            dx_tensor = dx_tensor.requires_grad_(True)
            C_values = model(dx_tensor)
            grad1 = torch.autograd.grad(C_values, dx_tensor, grad_outputs=torch.ones_like(C_values), create_graph=True)[0].item() * n
            grad2 = -grad1

            w1, w2 = constraint.p1.weight, constraint.p2.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            denominator = w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2) + alpha

            if denominator == 0:
                continue

            delta_lambda = -(C_values.item() + alpha * constraint.lambda_acc) / denominator
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

    def newtonian_substep(self):
        moving_particle = self.particles[3]
        fixed_particle = self.particles[2]
        spring_force = compute_spring_force(moving_particle.position, fixed_particle.position, 
                                         SPRING_CONSTANT, REST_LENGTH)

        moving_particle.velocity += ((spring_force + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        fixed_particle.velocity += ((-spring_force + fixed_particle.mass * self.gravity) / fixed_particle.mass) * self.sub_dt
        d = fixed_particle.velocity * self.sub_dt
        moving_particle.position -= d