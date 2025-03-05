# SimulatorNN.py
from Physics import *
import torch
import torch.nn as nn
import torch.nn.functional as F 

# Define the model
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


model = MLP()
checkpoint = torch.load('TrainedModels/gpu_muscle_length_and_act_best_model.pth')
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
            Constraint(self.particles[0], self.particles[1], MUSCLE_COMPLIANCE)
        ]

    def step(self, activation):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0

        for _ in range(self.num_substeps):
            self.xpbd_substep(activation)
            self.classic_substep(activation)
    #"""
    def xpbd_substep(self, activation):
        for particle in self.particles:
            if particle.xpbd:
                particle.prev_position = particle.position.copy()
                particle.prev_velocity = particle.velocity.copy()
                particle.velocity += self.gravity * self.sub_dt
                particle.position += particle.velocity * self.sub_dt

        for constraint in self.constraints:
            dx = -(constraint.p1.position - constraint.p2.position)
            n = dx / np.linalg.norm(dx)

            w1, w2 = constraint.p1.weight, constraint.p2.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)

            dx_tensor = torch.tensor(np.linalg.norm(dx)/params['lMopt'], dtype=torch.float32).reshape(-1, 1)
            activation = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
            inputs = torch.cat([dx_tensor, activation], dim=1).detach().requires_grad_(True)
            C_values = model(inputs)
            grad1 = n * torch.autograd.grad(C_values, inputs, grad_outputs=torch.ones_like(C_values), create_graph=True)[0][:, 0].item() 
            grad2 = -grad1
            denominator = (w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2)) + alpha
            if denominator == 0:
                continue
            delta_lambda = -C_values.item() / denominator
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

    def classic_substep(self, activation):
        moving_particle = self.particles[3]
        fixed_particle = self.particles[2]
        dx = fixed_particle.position - moving_particle.position
        lMtilde = np.linalg.norm(dx) / params['lMopt']
        spring_force = muscleForce(lMtilde, activation) * params['fMopt'] * normalized(dx)

        moving_particle.velocity += ((spring_force + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        fixed_particle.velocity += ((-spring_force + fixed_particle.mass * self.gravity) / fixed_particle.mass) * self.sub_dt
        d = fixed_particle.velocity * self.sub_dt
        moving_particle.position -= d