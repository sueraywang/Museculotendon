import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
import sys
sys.path.append('../python-nn')
from testMuscle import *

# Constants
m = 0.1  # Mass of particle (kg)
g = 9.8  # Gravity (m/s^2)
dt = 0.01  # Time step for simulation (seconds)

# Particle positions (x, y, z in meters)
lMuscle = 0.1  # Rest lengths of muscle (10 cm)
lTendon = 0.2  # Rest lengths of tendon (20 cm)
particle1_position = np.array([0.0, 0.0, 0.0])  # Fixed particle at origin
particle2_position = np.array([0.0, lMuscle + lTendon, 0.0])  # Moving particle
particle2_velocity = np.array([0.0, 0.0, 0.0])  # Moving particle
partition_point = np.array([lTendon, 0.0, 0.0]) # The "node" of muscle-tendon connection

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

def normalize(vec):
    return vec / np.linalg.norm(vec)

def calcVel(lM, lMT, act, params, curves):
    lMtilde = lM / params['lMopt']
    alphaM = 0.0
    lT = lMT - lM * np.cos(alphaM)
    lTtilde = lT / params['lTslack']
    vMtilde = calcVelTilde(lMtilde, lTtilde, act, params, curves)
    vM = vMtilde * params['lMopt'] * params['vMmax']
    if vM < 0 and lM < params['lMmin']:
        vM = 0
    return vM

# Function to update the position of the free particle
def update_particle():
    global particle2_position, particle2_velocity
    
    # Compute forces on particle 2
    force_gravity = np.array([0, -m * g * 0.01, 0])  # Gravity acts in the y-direction

    # Total force on particle 2
    total_force = force_gravity

    # Compute acceleration (F = ma)
    acceleration = total_force / m

    particle2_velocity += acceleration * dt

    particle2_position += particle2_velocity * dt

# Function to update the mulscle and tendon
def update_muscle():
    global particle2_position, lMuscle, lTendon, partition_point
    activation = 1.0
    alphaM = 0.0

    # Update velocity and position using Euler's method
    lMT = particle2_position - particle1_position
    vMuscle = float(calcVel(lMuscle, np.linalg.norm(lMT), activation, params, curves)) * params['lMopt'] * params['vMmax']
    lMuscle += vMuscle * dt
    lTendon = np.linalg.norm(lMT) - lMuscle * np.cos(alphaM)
    partition_point = normalize(lMT) * lTendon

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

    # Camera position
    gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0)

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

# Main simulation loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Update physics simulation
    update_particle()
    update_muscle()

    # Render the 3D scene
    render()

    # Sleep to simulate the time step
    time.sleep(dt)

# Cleanup
glfw.terminate()
