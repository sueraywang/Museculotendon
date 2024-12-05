import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
import sys
sys.path.append('../Museculotendon')
import matplotlib.pyplot as plt
from testMuscle import *

# Constants
mass = 0.1  # Mass of particle (kg)
weight = 1/mass
SUB_STEPS = 500
force_gravity = np.array([0, -mass * 9.8, 0])
dt = 0.01  # Time step for simulation (seconds)
k_spring = 10.0  # Spring constant
compliance = 1/k_spring
rest_length = 1.0

# Particle positions (x, y, z in meters)
particle1_position = np.array([0.5, 0.5, 0.0])  # Fixed particle at origin
previous_particle1_position = np.array([0.5, 0.5, 0.0])
particle1_velocity = np.array([0.0, 0.0, 0.0])
particle2_position = np.array([0.5, 0.0, 0.0])  # Moving particle
previous_particle2_position = np.array([0.5, 0.0, 0.0]) 
particle2_velocity = np.array([0.0, 0.0, 0.0])

# Newtonian ground truth particles
particle1_position_GT = np.array([-0.5, 0.5, 0.0])
particle1_velocity_GT = np.array([0.0, 0.0, 0.0])
particle2_position_GT = np.array([-0.5, 0.0, 0.0])
particle2_velocity_GT = np.array([0.0, 0.0, 0.0])

# Data storage for plotting
plot_time = 0
time_data = []
RL_over_time = []
RL_GT_over_time = []

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized!")
window = glfw.create_window(800, 600, "3D Muscle Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created!")
glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)

# Initialize Matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
line_x, = ax.plot([], [], 'r-', label="lM")
line_x1, = ax.plot([], [], 'b-', label="lM_GT")
ax.set_xlim(0, 10)
ax.set_ylim(-1.0, 1.0)
ax.set_title('Cubic Spring and Its Ground Truth Length')
ax.set_xlabel('time (s)')
ax.set_ylabel('Lengths (m)')
ax.legend(loc='upper right')

def normalized(vec):
    return vec / np.linalg.norm(vec)

def distance_constraint(compliance, dt):
    global particle1_position, particle2_position, weight
    alpha = compliance / (dt ** 2)
    w = 2 * weight
    grad = particle1_position - particle2_position
    length = np.linalg.norm(grad)
    if length == 0:
        return  # Prevent division by zero
    grad /= length
    C = (length - rest_length)
    delta_lagrangian_multiplier = -C / (w + alpha)
    particle1_position += grad * delta_lagrangian_multiplier * weight
    particle2_position += grad * -delta_lagrangian_multiplier * weight

def compute_spring_force(pos1, pos2, k, rest_len):
    displacement = pos1 - pos2
    length = np.linalg.norm(displacement)
    if length == 0:
        return np.array([0.0, 0.0, 0.0])
    force_magnitude = -k * (length - rest_len)
    return force_magnitude * (displacement / length)

def update_simulation():
    global particle1_position, particle2_position, particle1_velocity, particle2_velocity, previous_particle1_position, previous_particle2_position
    global particle1_position_GT, particle2_position_GT, particle1_velocity_GT, particle2_velocity_GT
    for _ in range(SUB_STEPS):
        # XPBD System
        particle1_velocity += force_gravity * (dt / SUB_STEPS) * weight
        particle2_velocity += force_gravity * (dt / SUB_STEPS) * weight
        previous_particle1_position = particle1_position.copy()
        previous_particle2_position = particle2_position.copy()
        particle1_position += particle1_velocity * dt / SUB_STEPS
        particle2_position += particle2_velocity * dt / SUB_STEPS
        distance_constraint(compliance, dt / SUB_STEPS)
        particle1_velocity = (particle1_position - previous_particle1_position) / (dt / SUB_STEPS)
        particle2_velocity = (particle2_position - previous_particle2_position) / (dt / SUB_STEPS)

        # Fix particle 1 for XPBD system
        d = np.array([0.5, 0.5, 0.0]) - particle1_position
        particle1_position += d
        particle2_position += d

        # Newtonian System (Ground Truth)
        spring_force_on_1 = compute_spring_force(particle1_position_GT, particle2_position_GT, k_spring, rest_length)
        spring_force_on_2 = -spring_force_on_1
        force_on_1 = spring_force_on_1 + force_gravity
        force_on_2 = spring_force_on_2 + force_gravity
        particle1_velocity_GT += (force_on_1 / mass) * (dt / SUB_STEPS)
        particle2_velocity_GT += (force_on_2 / mass) * (dt / SUB_STEPS)
        particle1_position_GT += particle1_velocity_GT * (dt / SUB_STEPS)
        particle2_position_GT += particle2_velocity_GT * (dt / SUB_STEPS)

        # Fix particle 1 for Newtonian system
        d_GT = np.array([-0.5, 0.5, 0.0]) - particle1_position_GT
        particle1_position_GT += d_GT
        particle2_position_GT += d_GT

def update_plot():
    global plot_time
    RL_over_time.append(np.linalg.norm(particle1_position - particle2_position))
    RL_GT_over_time.append(np.linalg.norm(particle1_position_GT - particle2_position_GT))
    time_data.append(plot_time)
    plot_time += dt
    line_x.set_data(time_data, RL_over_time)
    line_x1.set_data(time_data, RL_GT_over_time)
    ax.set_xlim(0, max(time_data) if time_data else 10)
    ax.set_ylim(min(min(RL_over_time), min(RL_GT_over_time)), max(max(RL_over_time), max(RL_GT_over_time)) if RL_over_time else 1.0)
    plt.draw()
    plt.pause(dt)

def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)
    glPushMatrix()
    glTranslatef(*particle1_position)
    glColor3f(1.0, 1.0, 0.0)
    draw_sphere(0.01, 20, 20)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(*particle2_position)
    glColor3f(0.0, 1.0, 0.0)
    draw_sphere(0.01, 20, 20)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(*particle1_position_GT)
    glColor3f(1.0, 1.0, 0.0)
    draw_sphere(0.01, 20, 20)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(*particle2_position_GT)
    glColor3f(0.0, 1.0, 0.0)
    draw_sphere(0.01, 20, 20)
    glPopMatrix()
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3fv(particle1_position)
    glVertex3fv(particle2_position)
    glEnd()
    glBegin(GL_LINES)
    glColor3f(0.5, 1.0, 1.0)
    glVertex3fv(particle1_position_GT)
    glVertex3fv(particle2_position_GT)
    glEnd()
    glfw.swap_buffers(window)

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

while not glfw.window_should_close(window):
    glfw.poll_events()
    update_simulation()
    update_plot()
    render()
    time.sleep(dt)

glfw.terminate()
