import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
import sys
sys.path.append('../Museculotendon')
from testMuscle import *
import matplotlib.pyplot as plt

# Constants
mass = 0.1  # Mass of particle (kg)
gravity = 9.8  # Gravity (m/s^2)
dt = 0.001  # Time step for simulation (seconds)
activation = 0.0
alphaMuscle = 0.0

def normalized(vec):
    return vec / np.linalg.norm(vec)

# Particle positions (x, y, z in meters)
lMuscle = 0.1  # Rest lengths of muscle (10 cm)
lTendon = 0.2  # Rest lengths of tendon (20 cm)
particle1_position = np.array([0.0, 0.0, 0.0])  # Fixed particle at origin
particle1_velocity = np.array([0.0, 0.0, 0.0])
particle2_position = normalized(np.array([0.0, -1.0, 0.0])) * (lMuscle + lTendon)  # Moving particle
particle2_velocity = np.array([0.0, 0.0, 0.0])
partition_point = normalized(particle2_position - particle1_position) * lTendon # The "node" of muscle-tendon connection

# Camera control parameters (spherical coordinates)
camera_radius = 1.0
camera_azimuth = 0.0  # Horizontal rotation (radians)
camera_elevation = math.radians(30)  # Vertical rotation (radians)
camera_speed = 0.05  # Speed of zooming

# Mouse tracking variables
scroll_sensitivity = 0.2  # Sensitivity of mouse scroll zooming

# Data storage for plotting
time_data = []
plot_time = 0
lM_over_time = []
lT_over_time = []
lMT_over_time = []
activation_over_time = []

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
fig, axs = plt.subplots(2, 1, figsize=(6, 8))
# Setup each subplot for x, y, and z positions
line_x, = axs[0].plot([], [], 'r-', label="lM")
line_x1, = axs[0].plot([], [], 'g-', label="lT")
line_x2, = axs[0].plot([], [], 'b-', label="lMT")
axs[0].set_xlim(0, 10)
axs[0].set_ylim(-1.0, 1.0)
axs[0].set_title('Muscle and Tendon Length')
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('Lengths (m)')
axs[0].legend()

line_y, = axs[1].plot([], [], 'y-', label="activation")
axs[1].set_xlim(0, 10)
axs[1].set_ylim(0, 1.0)
axs[1].set_title('Activation')
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('act')

def computeVelTilde(lMtilde, lTtilde, act, alphaM, params, curves):
    cosAlphaM = np.cos(alphaM)
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    tfl = tfl = curves['TFL'].calcValue(lTtilde)
    vMtildeInit = 0.0

    # Use fsolve to compute the muscle fiber velocity
    vMtilde = fzero_newton(lambda vMtilde: forceBalance(vMtilde, act, afl, pfl, tfl, curves['FV'], cosAlphaM, params), vMtildeInit, verbose=False)[0]
    return vMtilde

def computeVel(lM, lMT, act, alphaM, params, curves):
    lMtilde = lM / params['lMopt']
    lT = lMT - lM * np.cos(alphaM)
    lTtilde = lT / params['lTslack']
    vMtilde = computeVelTilde(lMtilde, lTtilde, act, alphaM, params, curves)
    vM = vMtilde * params['lMopt'] * params['vMmax']
    
    return vM

def muscleForce(vM, lM, act, alphaM, params):
    lMtilde = lM / params['lMopt']
    vMtilde = vM / params['lMopt'] / params['vMmax']
    afl = curves['AFL'].calcValue(lMtilde)
    pfl = curves['PFL'].calcValue(lMtilde)
    deriv = curves['FV'].calcValDeriv(vMtilde)
    fv = deriv[0]
    fM = act * afl * fv + pfl + params['beta'] * vMtilde
    f = fM * np.cos(alphaM) * params['fMopt']
    return f

def applyConstraint():
    global particle1_position, particle2_position, particle1_velocity, particle2_velocity, lMuscle

    d = np.array([0.0, 0.0, 0.0]) - particle1_position
    particle1_position += d
    particle2_position += d
    #particle1_velocity = np.array([0.0, 0.0, 0.0])
    #particle2_velocity = d/dt

# Function to update the mulscle and tendon
def update_muscle():
    global particle1_position, particle2_position, particle1_velocity, particle2_velocity, lMuscle

    # Update muscle velocity and length given by particles
    lMT_current = -(particle2_position - particle1_position)
    vMuscle = computeVel(lMuscle, np.linalg.norm(lMT_current), activation, alphaMuscle, params, curves)
    lMuscle += float(vMuscle) * dt

    # Compute forces
    force_gravity = np.array([0, -mass * gravity, 0])  # Gravity acts in the y-direction
    force_muscle = muscleForce(vMuscle, lMuscle, activation, alphaMuscle, params) * normalized(lMT_current)
    total_force = force_muscle + force_gravity

    # Compute acceleration and update particles' positions
    acceleration_1 = -total_force / mass
    particle1_velocity += acceleration_1 * dt
    particle1_position += particle1_velocity * dt

    acceleration_2 = total_force / mass
    particle2_velocity += acceleration_2 * dt
    particle2_position += particle2_velocity * dt

    applyConstraint()

    #print(force_muscle, total_force)

def update_partition_point():
    global particle1_position, particle2_position, lMuscle, lTendon, partition_point

    lMT_next = -(particle2_position - particle1_position)
    lTendon = np.linalg.norm(lMT_next) - lMuscle * np.cos(alphaMuscle)
    partition_point = -lTendon * normalized(lMT_next)

def update_plot():
    global lMuscle, lTendon, activation, plot_time
    lM_over_time.append(lMuscle)
    lT_over_time.append(lTendon)
    lMT_over_time.append(lTendon + lMuscle)
    activation_over_time.append(activation)
    time_data.append(plot_time)
    plot_time += dt
    
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

    # Render moving particle (Particle 2)
    glPushMatrix()
    glTranslatef(*particle2_position)
    glColor3f(0.0, 1.0, 0.0)  # Green color
    draw_sphere(0.01, 20, 20)  # Draw particle 2 as a small sphere
    glPopMatrix()

    # Render muscle and tendon (as lines)
    glColor3f(1.0, 0.0, 0.0)  # Red color for tendon
    glBegin(GL_LINES)
    glVertex3fv(particle1_position)
    glVertex3fv(partition_point)
    glEnd()
    glColor3f(1.0, 1.0, 1.0)  # white color for muscle
    glBegin(GL_LINES)
    glVertex3fv(partition_point)
    glVertex3fv(particle2_position)
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

    # Update activation
    activation = np.sin(plot_time * 20) / 2.0 + 0.5

    # Update physics simulation
    update_muscle()
    update_partition_point()
    update_plot()

    # Update the real-time plot
    line_x.set_data(time_data, lM_over_time)
    line_x1.set_data(time_data, lT_over_time)
    line_x2.set_data(time_data, lMT_over_time)
    line_y.set_data(time_data, activation_over_time)
    # Adjust x-axis dynamically for each subplot
    for ax in axs:
        ax.set_xlim(0, max(time_data))  # Adjust time window
    # Adjust y-axis dynamically for each subplot
    axs[0].set_ylim(min(lM_over_time), max(lMT_over_time))
    axs[1].set_ylim(min(activation_over_time), max(activation_over_time))
    plt.draw()
    plt.pause(dt)

    # Render the 3D scene
    render()

    # Sleep to simulate the time step
    time.sleep(dt)

# Cleanup
glfw.terminate()