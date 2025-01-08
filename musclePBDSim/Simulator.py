# Simulator.py
# Constains Euler and XPBD integration
from Physics import *

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
            dx = constraint.p1.position - constraint.p2.position
            n = dx / np.linalg.norm(dx)
            dx_prev = constraint.p1.prev_position-constraint.p2.prev_position
            displacement = np.linalg.norm(dx) - REST_LENGTH
            
            C = displacement**3#2 *(1/np.sqrt(2))
            grad1 = n# * np.sqrt(2) * displacement
            grad2 = -grad1

            w1, w2 = constraint.p1.weight, constraint.p2.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            beta = DAMPING_CONSTANT * self.sub_dt**2
            gamma = (alpha * beta)/self.sub_dt 

            denominator = (1 + gamma) * (w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2)) + alpha
            if denominator == 0:
                continue
            delta_lambda = -(C + alpha * constraint.lambda_acc + gamma * np.dot(grad1, dx - dx_prev)) / denominator
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