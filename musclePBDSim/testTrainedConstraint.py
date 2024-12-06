import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
import sys
import torch
import torch.nn as nn
sys.path.append('../Museculotendon')
import matplotlib.pyplot as plt
from testMuscle import *

# Constants
mass = 0.1  # Mass of particle (kg)
force_gravity = np.array([0, -mass * 9.8, 0])
dt = 0.01  # Time step for simulation (seconds)
activation = 1.0
alphaMuscle = 0.0
compliance = 1/params['fMopt']/params['lMopt']       # Compliance value to control stiffness

# Particle positions (x, y, z in meters)
lMuscle = 0.1  # Rest lengths of muscle (10 cm)
particle1_position = np.array([0.5, 0.0, 0.0])  # Fixed particle at origin
particle1_velocity = np.array([0.0, 0.0, 0.0])
particle2_position = np.array([0.5, -lMuscle, 0.0])  # Moving particle
particle2_velocity = np.array([0.0, 0.0, 0.0])

lMuscle_GT = 0.1
particle1_position_GT = np.array([-0.5, 0.0, 0.0])  # Fixed particle at origin
particle1_velocity_GT = np.array([0.0, 0.0, 0.0])
particle2_position_GT = np.array([-0.5, -lMuscle_GT, 0.0])  # Moving particle
particle2_velocity_GT = np.array([0.0, 0.0, 0.0])

# Camera control parameters (spherical coordinates)
camera_radius = 2.0
camera_azimuth = 0.0  # Horizontal rotation (radians)
camera_elevation = math.radians(0)  # Vertical rotation (radians)
camera_speed = 0.05  # Speed of zooming

# Mouse tracking variables
scroll_sensitivity = 0.2  # Sensitivity of mouse scroll zooming

# Define the MLP model for C(l_tilde)
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, layers=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        for _ in range(layers - 1):
            self.model.append(nn.Linear(hidden_size, hidden_size))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(hidden_size, output_size))

    def forward(self, l_tilde):
        # Adjusted for single input
        return self.model(l_tilde.unsqueeze(-1)).squeeze(-1)


model = MLP(hidden_size=128, layers=6)
model.load_state_dict(torch.load('musclePBDSim/forceConstraint.pth'))
model.eval()

# Data storage for plotting
plot_time = 0
time_data = []
lM_over_time = []
lM_GT_over_time = []

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

window = glfw.create_window(800, 600, "3D Muscle Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created!")

glfw.make_context_current(window)

# Enable depth test
glEnable(GL_DEPTH_TEST)

# Initialize Matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))  # Only one plot is required
line_x, = ax.plot([], [], 'r-', label="lM")
line_x1, = ax.plot([], [], 'b-', label="lM_GT")
ax.set_xlim(0, 10)
ax.set_ylim(-1.0, 1.0)
ax.set_title('Muscle and Its Ground Truth Length')
ax.set_xlabel('time (s)')
ax.set_ylabel('Lengths (m)')
ax.legend(loc='upper right')  # Fix legend position to upper right


def normalized(vec):
    return vec / np.linalg.norm(vec)

def applyConstraint():
    global lMuscle
    lMtilde = lMuscle / params['lMopt']
    lMtilde_tensor = torch.tensor(lMtilde, dtype=torch.float32)
    lMtilde_tensor = lMtilde_tensor.requires_grad_(True)
    C_values = model(lMtilde_tensor)
    f = -(C_values * torch.autograd.grad(C_values, lMtilde_tensor, grad_outputs=torch.ones_like(C_values), create_graph=True)[0]).item() * params['fMopt']
    return f

def update_muscle():
    global particle1_position, particle2_position, particle1_velocity, particle2_velocity, lMuscle
    
    lMT_current = -(particle2_position - particle1_position)
    lMuscle = np.linalg.norm(lMT_current)
    force_muscle = applyConstraint() * normalized(lMT_current)

    # Total force on particle 2
    total_force = force_muscle + force_gravity

    # Compute acceleration and update particles' positions
    acceleration_1 = -total_force / mass
    particle1_velocity += acceleration_1 * dt
    particle1_position += particle1_velocity * dt

    acceleration_2 = total_force / mass
    particle2_velocity += acceleration_2 * dt
    particle2_position += particle2_velocity * dt

    # "Fix" particle 1
    d = np.array([0.5, 0.0, 0.0]) - particle1_position
    particle1_position += d
    particle2_position += d

def muscleForce(vM, lM, act, alphaM, params):
    lMtilde = lM / params['lMopt']
    vMtilde = vM / params['lMopt'] / params['vMmax']
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    deriv = curves['FV'].calcValDeriv(vMtilde)
    fv = deriv[0]
    fM = act * afl + pfl
    f = fM * np.cos(alphaM) * params['fMopt']
    return f

def update_muscle_GT():
    global particle1_position_GT, particle2_position_GT, particle1_velocity_GT, particle2_velocity_GT, lMuscle_GT

    # Update muscle velocity and length given by particles
    lMT_current = -(particle2_position_GT - particle1_position_GT)
    lMuscle_GT = np.linalg.norm(lMT_current)
    vMuscle = np.array([-np.linalg.norm(particle2_velocity_GT - particle1_velocity_GT)]) if (lMuscle_GT < 0.1) else np.array([np.linalg.norm(particle2_velocity_GT - particle1_velocity_GT)])

    # Compute forces on particle 2
    force_muscle = muscleForce(vMuscle, lMuscle_GT, activation, alphaMuscle, params) * normalized(lMT_current)

    # Total force on particle 2
    total_force = force_muscle + force_gravity

    # Compute acceleration and update particles' positions
    acceleration_1 = -total_force / mass
    particle1_velocity_GT += acceleration_1 * dt
    particle1_position_GT += particle1_velocity_GT * dt

    acceleration_2 = total_force / mass
    particle2_velocity_GT += acceleration_2 * dt
    particle2_position_GT += particle2_velocity_GT * dt

    # "Fix" particle 1
    d = np.array([-0.5, 0.0, 0.0]) - particle1_position_GT
    particle1_position_GT += d
    particle2_position_GT += d

def update_plot():
    global lMuscle, plot_time, lMuscle_GT
    lM_over_time.append(lMuscle)
    lM_GT_over_time.append(lMuscle_GT)
    time_data.append(plot_time)
    plot_time += dt
    line_x.set_data(time_data, lM_over_time)
    line_x1.set_data(time_data, lM_GT_over_time)

    # Adjust axes as needed
    ax.set_xlim(0, max(time_data) if time_data else 10)
    ax.set_ylim(min(min(lM_over_time), min(lM_GT_over_time)), max(max(lM_over_time), max(lM_GT_over_time)) if lM_over_time else 1.0)

    plt.draw()
    plt.pause(dt)

# Function to render the sphere using triangles
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

# Function to render the scene
def render():
    # Clear the screen and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Setup the 3D view
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Compute the camera position using spherical coordinates
    camera_x = camera_radius * math.cos(camera_elevation) * math.sin(camera_azimuth)
    camera_y = camera_radius * math.sin(camera_elevation)
    camera_z = camera_radius * math.cos(camera_elevation) * math.cos(camera_azimuth)

    # Camera looking at the origin (0, 0, 0)
    gluLookAt(camera_x, camera_y, camera_z, 0, 0, 0, 0, 1, 0)

    # Render fixed particle (Particle 1)
    glPushMatrix()
    glTranslatef(*particle1_position)
    glColor3f(1.0, 1.0, 0.0)  # Yellow color
    draw_sphere(0.01, 20, 20)  # Draw particle 1 as a small sphere
    glPopMatrix()
    glPushMatrix()
    glTranslatef(*particle2_position)
    glColor3f(0.0, 1.0, 0.0)  # Green color
    draw_sphere(0.01, 20, 20)  # Draw particle 2 as a small sphere
    glPopMatrix()

    # Render ground truth
    glPushMatrix()
    glTranslatef(*particle1_position_GT)
    glColor3f(1.0, 1.0, 0.0)  # Yellow color
    draw_sphere(0.01, 20, 20)  # Draw particle 1 as a small sphere
    glPopMatrix()
    glPushMatrix()
    glTranslatef(*particle2_position_GT)
    glColor3f(0.0, 1.0, 0.0)  # Green color
    draw_sphere(0.01, 20, 20)  # Draw particle 2 as a small sphere
    glPopMatrix()

    # Render muscle and tendon (as lines)
    glColor3f(1.0, 0.0, 0.0)  # red color for muscle
    glBegin(GL_LINES)
    glVertex3fv(particle1_position)
    glVertex3fv(particle2_position)
    glEnd()

    glColor3f(0.5, 1.0, 1.0)  # blue for GT
    glBegin(GL_LINES)
    glVertex3fv(particle1_position_GT)
    glVertex3fv(particle2_position_GT)
    glEnd()

    glfw.swap_buffers(window)

# Callback for mouse scroll (zoom in/out)
def scroll_callback(window, xoffset, yoffset):
    global camera_radius
    camera_radius -= yoffset * scroll_sensitivity
    if camera_radius < 0.5:
        camera_radius = 0.5  # Prevent zooming too close

# Callback for keyboard input (WASD for camera rotation)
def key_callback(window, key, scancode, action, mods):
    global camera_azimuth, camera_elevation
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_W:
            camera_elevation += camera_speed  # Rotate camera up
        elif key == glfw.KEY_S:
            camera_elevation -= camera_speed  # Rotate camera down
        elif key == glfw.KEY_A:
            camera_azimuth -= camera_speed  # Rotate camera left
        elif key == glfw.KEY_D:
            camera_azimuth += camera_speed  # Rotate camera right

        # Clamp the elevation angle to avoid flipping the view
        camera_elevation = max(-math.pi / 2, min(math.pi / 2, camera_elevation))

# Register scroll and keyboard callbacks
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_key_callback(window, key_callback)

# Main simulation loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    
    # Update physics simulation
    update_muscle()
    update_muscle_GT()
    update_plot()

    # Render the 3D scene
    render()

    # Sleep to simulate the time step
    time.sleep(dt)

# Cleanup
glfw.terminate()