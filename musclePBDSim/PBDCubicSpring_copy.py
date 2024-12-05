import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time
import sys
sys.path.append('../Museculotendon')
from testMuscle import *
import matplotlib.pyplot as plt

# Constants
mass = 0.1  # Mass of particle (kg)
SUB_STEPS = 500
acc_gravity = np.array([0, -9.8, 0])
dt = 1/60  # Time step for simulation (seconds)
k_spring = 0.17  # Spring constant
compliance = 1/k_spring
rest_length = 1.4
initial_length = 0.1
xpbd_fix_pos = np.array([0.5, 0.5, 0.0])
newtonian_fix_pos = np.array([-0.5, 0.5, 0.0])

xpbd_free_pos = xpbd_fix_pos + np.array([0.0, -initial_length, 0.0])
newtonian_free_pos = newtonian_fix_pos + np.array([0.0, -initial_length, 0.0])

def normalized(vec):
    return vec / np.linalg.norm(vec)
    
def compute_spring_force(pos1, pos2, k, rest_len):
    displacement = pos1 - pos2
    length = np.linalg.norm(displacement)
    if length == 0:
        return np.array([0.0, 0.0, 0.0])
    force_magnitude = -k * (length - rest_len)**3
    return force_magnitude * normalized(displacement)

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()

class Particle:
    def __init__(self, position, mass=0.1, fixed=False, xpbd=False):
        self.position = position
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(3)
        self.mass = mass
        self.weight = 0.0 if fixed else 1.0 / mass
        self.fixed = fixed
        self.xpbd = xpbd

class Constraint:
    def __init__(self, p1, p2, compliance):
        self.p1 = p1
        self.p2 = p2
        self.compliance = compliance
        self.lambda_acc = 0.0

class Simulator:
    def __init__(self):
        self.dt = dt
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.gravity = acc_gravity

        # Initialize particles with a larger vertical separation
        self.particles = [
            #XPBD
            Particle(xpbd_fix_pos.copy(), fixed = True, xpbd=True),   
            Particle(xpbd_free_pos.copy(), xpbd=True),
            #Newtonian
            Particle(newtonian_fix_pos.copy(), fixed = True),
            Particle(newtonian_free_pos.copy())
        ]

        self.constraints = [
            Constraint(
                self.particles[0], self.particles[1], compliance)
        ]

    def step(self):
        for constraint in self.constraints:
            constraint.lambda_acc = 0.0

        for _ in range(self.num_substeps):
            self.xpbd_substep()
            self.newtonian_substep()

    def xpbd_substep(self):
        # Predicted positions
        for particle in self.particles:
            if particle.xpbd and not particle.fixed:
                particle.prev_position = particle.position.copy()
                particle.velocity += self.gravity * self.sub_dt
                particle.position += particle.velocity * self.sub_dt

        for constraint in self.constraints:
            x1 = constraint.p1.position
            x2 = constraint.p2.position

            diff = x1 - x2
            current_length = np.linalg.norm(diff)

            if current_length < 1e-7:
                continue

            displacement = current_length - rest_length
            n = diff / current_length  # Normalized vector

            # Energy-based constraint
            C = 1/ np.sqrt(2) * displacement ** 2
            grad1 = n * displacement * np.sqrt(2) # * 3 * displacement ** 2
            grad2 = -n * displacement * np.sqrt(2) # * 3 * displacement ** 2

            w1 = constraint.p1.weight
            w2 = constraint.p2.weight

            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            denominator = w1 * np.dot(grad1, grad1) + w2 * np.dot(grad2, grad2) + alpha

            if denominator == 0:
                continue

            delta_lambda = -(C + alpha * constraint.lambda_acc) / denominator
            #constraint.lambda_acc += delta_lambda

            constraint.p1.position += w1 * delta_lambda * grad1
            constraint.p2.position += w2 * delta_lambda * grad2

        for particle in self.particles:
            if particle.xpbd and not particle.fixed:
                particle.velocity = (particle.position - particle.prev_position) / self.sub_dt

                
    def newtonian_substep(self):
        # Compute the spring force on the moving particle (particle 3)
        spring_force = compute_spring_force(self.particles[3].position, self.particles[2].position, k_spring, rest_length)
        total_force = spring_force + self.particles[3].mass * self.gravity

        # Update velocity and position of the moving particle
        self.particles[3].velocity += (total_force / self.particles[3].mass) * self.sub_dt
        self.particles[3].position += self.particles[3].velocity * self.sub_dt
            
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

        self.window = glfw.create_window(self.width, self.height, "Cubic Spring Simulation", None, None)
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
        gluLookAt(0, 0, 3,  # Camera position
                  0, 0, 0,  # Look at point
                  0, 1, 0)  # Up vector

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw coordinate axes for reference
        glBegin(GL_LINES)
        # X axis (white)
        glColor3f(1.0, 1.0, 1.0)
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
        for particle in simulator.particles:
            if particle.fixed:
                glColor3f(1.0, 0.0, 0.0)  # Red for fixed particle
            else:
                glColor3f(0.0, 1.0, 0.0)  # Green for moving particle
            glPushMatrix()
            glTranslatef(*(particle.position))
            draw_sphere(0.05, 20, 20)
            glPopMatrix()

            # Print position for debugging
            #print(f"Particle position: {particle.position}")

        # Draw spring with thicker line
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(*self.simulator.particles[0].position)
        glVertex3f(*self.simulator.particles[1].position)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.5, 1.0, 1.0)
        glVertex3f(*self.simulator.particles[2].position)
        glVertex3f(*self.simulator.particles[3].position)
        glEnd()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        glfw.terminate()

# Data storage for plotting
plot_time = 0
time_data = []
xpbd_over_time = []
newtownian_over_time = []
simulator = Simulator()
renderer = OpenGLRenderer(simulator)

# Initialize Matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
line_x, = ax.plot([], [], 'r-', label="XPBD Current Length")
line_x1, = ax.plot([], [], 'b-', label="Newtonian Current Length")
ax.set_xlim(0, 10)
ax.set_ylim(-1.0, 1.0)
ax.set_title('Cubic Spring Length')
ax.set_xlabel('time (s)')
ax.set_ylabel('Lengths (m)')
ax.legend(loc='upper right')

def update_plot():
    global plot_time, time_data
    xpbd_over_time.append(np.linalg.norm(simulator.particles[0].position - simulator.particles[1].position))
    newtownian_over_time.append(np.linalg.norm(simulator.particles[2].position - simulator.particles[3].position))
    time_data.append(plot_time)
    plot_time += dt
    line_x.set_data(time_data, xpbd_over_time)
    line_x1.set_data(time_data, newtownian_over_time)
    ax.set_xlim(0, max(time_data) if time_data else 10)
    ax.set_ylim(min(min(xpbd_over_time), min(newtownian_over_time)), max(max(xpbd_over_time), max(newtownian_over_time)) if xpbd_over_time else 1.0)
    plt.draw()
    plt.pause(dt)

while not renderer.should_close():
    glfw.poll_events()
    simulator.step()
    renderer.render()
    update_plot()
    time.sleep(dt)

renderer.cleanup()