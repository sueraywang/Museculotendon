import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import time

class Particle:
    def __init__(self, position, mass=1.0, fixed=False):
        self.position = np.array(position, dtype=np.float32)
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.mass = mass
        self.inv_mass = 0.0 if fixed else 1.0/mass
        self.fixed = fixed

class SpringConstraint:
    def __init__(self, p1, p2, rest_length, stiffness):
        self.p1 = p1
        self.p2 = p2
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.compliance = 1 / stiffness
        self.lambda_acc = 0.0

class XPBDSimulator:
    def __init__(self):
        self.dt = 1/60.0
        self.num_substeps = 1
        self.sub_dt = self.dt / self.num_substeps
        self.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        
        # Initialize particles with a larger vertical separation
        self.particles = [
            Particle([0.0, 1.0, 0.0], fixed=True),    # Fixed particle at top
            Particle([0.0, 0.0, 0.0], mass=0.1)       # Moving particle below
        ]
        
        rest_length = 0.5
        stiffness = 100.0
        self.constraints = [
            SpringConstraint(self.particles[0], self.particles[1], rest_length, stiffness)
        ]

    def step(self):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0
            
        for _ in range(self.num_substeps):
            self._substep()

    def _substep(self):
        for particle in self.particles:
            if not particle.fixed:
                particle.prev_position = particle.position.copy()
                particle.velocity += self.gravity * self.sub_dt
                particle.position += particle.velocity * self.sub_dt

        num_iterations = 4
        for _ in range(num_iterations):
            for constraint in self.constraints:
                x1 = constraint.p1.position
                x2 = constraint.p2.position
                
                diff = x2 - x1
                current_length = np.linalg.norm(diff)
                
                if current_length < 1e-7:
                    continue
                
                C = current_length - constraint.rest_length
                n = diff / current_length
                grad1 = -n
                grad2 = n
                
                w1 = constraint.p1.inv_mass
                w2 = constraint.p2.inv_mass
                denominator = w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2)
                
                if denominator == 0:
                    continue
                
                alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
                delta_lambda = -(C + alpha * constraint.lambda_acc) / (denominator + alpha)
                constraint.lambda_acc += delta_lambda
                
                if not constraint.p1.fixed:
                    constraint.p1.position += w1 * delta_lambda * grad1
                if not constraint.p2.fixed:
                    constraint.p2.position += w2 * delta_lambda * grad2

        for particle in self.particles:
            if not particle.fixed:
                particle.velocity = (particle.position - particle.prev_position) / self.sub_dt

class OpenGLRenderer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.width = 800
        self.height = 600

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Set OpenGL version and profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
        self.window = glfw.create_window(self.width, self.height, "Spring Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        
        # Basic OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)  # Dark gray background
        
        # Set up viewport and projection matrix
        glViewport(0, 0, self.width, self.height)
        self.setup_camera()

    def setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width/self.height, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Move camera back and slightly up
        gluLookAt(0, 1, 5,  # Camera position
                  0, 0, 0,  # Look at point
                  0, 1, 0)  # Up vector

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 1, 5, 0, 0, 0, 0, 1, 0)

        # Draw coordinate axes for reference
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        
        # Draw particles with larger size
        glPointSize(15.0)
        glBegin(GL_POINTS)
        for particle in self.simulator.particles:
            if particle.fixed:
                glColor3f(1.0, 0.0, 0.0)  # Red for fixed particle
            else:
                glColor3f(0.0, 1.0, 0.0)  # Green for moving particle
            glVertex3f(*particle.position)
            
            # Print position for debugging
            print(f"Particle position: {particle.position}")
        glEnd()
        
        # Draw spring with thicker line
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)  # White spring
        glVertex3f(*self.simulator.particles[0].position)
        glVertex3f(*self.simulator.particles[1].position)
        glEnd()
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        glfw.terminate()

def main():
    print("Starting simulation...")  # Debug print
    simulator = XPBDSimulator()
    renderer = OpenGLRenderer(simulator)
    
    last_time = time.time()
    
    print("Entering main loop...")  # Debug print
    while not renderer.should_close():
        current_time = time.time()
        elapsed = current_time - last_time
        
        if elapsed >= simulator.dt:
            simulator.step()
            last_time = current_time
        
        renderer.render()
    
    renderer.cleanup()
    print("Simulation ended.")  # Debug print

if __name__ == "__main__":
    main()