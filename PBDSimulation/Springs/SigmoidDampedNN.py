# Simulator.py
from Physics import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

def compute_sigmoid_damped_force(pos1, pos2, vel1, vel2, k, rest_len):
    displacement = pos1 - pos2
    length = np.linalg.norm(displacement)
    if length == 0:
        return np.array([0.0, 0.0, 0.0])
    force_magnitude = -k * (length - rest_len)**3 / (1 + np.exp(vel1 - vel2))
    return force_magnitude * normalized(displacement)

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
checkpoint = torch.load('TrainedModels/Springs/cubic_spring_model.pth')
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
            Constraint(self.particles[0], self.particles[1], SPRING_COMPLIANCE)
        ]

    def step(self):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0

        for _ in range(self.num_substeps):
            self.xpbd_substep()
            self.classic_substep()

    def xpbd_substep(self):
        fixed_particle = self.particles[0]
        moving_particle = self.particles[1]
        moving_particle.prev_position = moving_particle.position.copy()
        moving_particle.velocity += self.gravity * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt

        for constraint in self.constraints:
            dx = fixed_particle.position - moving_particle.position
            dx_prev = fixed_particle.prev_position-moving_particle.prev_position
            relative_motion = dx - dx_prev
            n = dx / np.linalg.norm(dx)
            dx_tensor = torch.tensor(np.linalg.norm(dx)-REST_LENGTH, dtype=torch.float32).reshape(-1, 1)
            inputs = torch.cat([dx_tensor], dim=1).detach().requires_grad_(True)
            C = model(inputs)
            C_values = model(inputs).item() / (1 + np.exp(moving_particle.velocity - fixed_particle.velocity))
            grad = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 0].item()

            w1, w2 = fixed_particle.weight, moving_particle.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            beta = DAMPING_CONSTANT * self.sub_dt**2

            denominator = (w1 * np.dot(grad, grad) + w2 * np.dot(grad, grad)) + alpha
            if denominator == 0:
                continue
            delta_lambda = -C_values / denominator
            fixed_particle.position += w1 * delta_lambda * grad * n
            moving_particle.position -= w2 * delta_lambda * grad * n 
            
            """
            # Update damping
            relative_vel = fixed_particle.velocity - moving_particle.velocity
            grad = relative_vel / np.linalg.norm(relative_vel)
            delta_lambda_damping = -(beta/self.sub_dt * np.dot(grad, relative_motion)) / (beta/self.sub_dt * (w1 * np.dot(grad, grad) + w2 * np.dot(grad, grad)) + 1)
            fixed_particle.position += w1 * delta_lambda_damping * grad
            moving_particle.position -= w2 * delta_lambda_damping * grad
            """ 

            moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
            
            d = fixed_particle.position - fixed_particle.prev_position
            fixed_particle.position -= d
            moving_particle.position -= d

    def classic_substep(self):
        moving_particle = self.particles[3]
        fixed_particle = self.particles[2]
        
        # Apply damping force along constraint direction
        spring_force = compute_sigmoid_damped_force(moving_particle.position, fixed_particle.position, moving_particle.velocity, fixed_particle.velocity,
                                         SPRING_CONSTANT, REST_LENGTH)
        
        # Update velocity and position
        moving_particle.velocity += ((spring_force + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt