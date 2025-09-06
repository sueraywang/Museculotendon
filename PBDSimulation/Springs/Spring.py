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
            Constraint(self.particles[0], self.particles[1], SPRING_COMPLIANCE)
        ]

    def step(self):
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
            n = dx / np.linalg.norm(dx)
            displacement = np.linalg.norm(dx) - REST_LENGTH
            C = displacement**2 *(1/np.sqrt(2))
            grad = np.sqrt(2) * displacement

            w1, w2 = fixed_particle.weight, moving_particle.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)

            denominator = (w1 * np.dot(grad, grad) + w2 * np.dot(grad, grad)) + alpha
            if denominator == 0:
                continue
            delta_lambda = -C / denominator
            fixed_particle.position += w1 * delta_lambda * grad * n
            moving_particle.position -= w2 * delta_lambda * grad * n 
            
            moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
            
            d = fixed_particle.position - fixed_particle.prev_position
            fixed_particle.position -= d
            moving_particle.position -= d 

    def classic_substep(self):
        moving_particle = self.particles[3]
        fixed_particle = self.particles[2]
        spring_force = compute_spring_force(moving_particle.position, fixed_particle.position, 
                                         SPRING_CONSTANT, REST_LENGTH)

        moving_particle.velocity += ((spring_force + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt