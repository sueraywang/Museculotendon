import glfw
from OpenGL.GL import *
import numpy as np

# Simulation parameters
dt = 0.01
STEPS = 500
mass = 0.1  # Mass of particle (kg)
weight = 1 / mass

# Particle properties
particle1_position = np.array([0.0, 0.0, 0.0])  # Fixed particle
particle1_velocity = np.array([0.0, 0.0, 0.0])
particle2_position = np.array([0.0, -2.0, 0.0])  # Moving particle
previous_particle2_position = np.array([0.0, -2.0, 0.0])
particle2_velocity = np.array([0.0, 0.0, 0.0])

# External force
force_gravity = np.array([0, -mass * 9.8, 0])

def distance_constraint(compliance, dt):
    global particle1_position, particle2_position, weight
    alpha = compliance / (dt ** 2)
    w = 2 * weight

    # Calculate the gradient vector and normalize
    grad = particle1_position - particle2_position
    length = np.linalg.norm(grad)
    if length == 0:
        return  # Prevent division by zero

    grad /= length
    rest_len = 1
    C = length - rest_len
    delta_lagrangian_multiplier = -C / (w + alpha)

    # Apply distance constraint
    particle1_position += grad * delta_lagrangian_multiplier * weight
    particle2_position += grad * -delta_lagrangian_multiplier * weight

def update_particles():
    global particle2_velocity, particle2_position, previous_particle2_position, particle1_position
    for _ in range(STEPS):
        # Apply gravity to particle2's velocity
        particle2_velocity += force_gravity * (dt / STEPS) * weight

        # Store the previous position of particle2 before updating
        previous_particle2_position = particle2_position.copy()

        # Update predicted position
        particle2_position += particle2_velocity * dt / STEPS

        # Apply the distance constraint to maintain the distance between particles
        distance_constraint(0.01, dt / STEPS)

        # Update the velocity based on the change in position
        particle2_velocity = (particle2_position - previous_particle2_position) / (dt / STEPS)

        # "Fix" particle 1
    d = np.array([0.0, 0.0, 0.0]) - particle1_position
    particle1_position += d
    particle2_position += d
    previous_particle2_position += d

def draw_particles():
    glClear(GL_COLOR_BUFFER_BIT)

    glPointSize(10)
    glBegin(GL_POINTS)
    glVertex3f(*particle1_position)
    glVertex3f(*particle2_position)
    glEnd()

    glBegin(GL_LINES)
    glVertex3f(*particle1_position)
    glVertex3f(*particle2_position)
    glEnd()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(640, 480, "Particle Simulation", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Set up the OpenGL environment
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-5, 5, -5, 5, -1, 1)  # Set up orthographic projection
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    while not glfw.window_should_close(window):
        update_particles()
        
        # Render particles
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_particles()
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
