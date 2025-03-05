# Renderer.py
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import math

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

class OpenGLRenderer:
    def __init__(self, simulator, width=800, height=600):
        self.simulator = simulator
        self.width = width
        self.height = height
        self._init_glfw()
        self._setup_gl()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        self.window = glfw.create_window(self.width, self.height, 
                                       "Muscle Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

    def _setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glViewport(0, 0, self.width, self.height)
        self._setup_camera()

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width/self.height, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)

    def _draw_axes(self):
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

    def _draw_particles(self):
        for particle in self.simulator.particles:
            glColor3f(1.0, 0.0, 0.0) if particle.fixed else glColor3f(0.0, 1.0, 0.0)
            glPushMatrix()
            glTranslatef(*particle.position)
            draw_sphere(0.05, 20, 20)
            glPopMatrix()

    def _draw_springs(self):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # XPBD spring
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(*self.simulator.particles[0].position)
        glVertex3f(*self.simulator.particles[1].position)
        # Newtonian spring
        glColor3f(0.5, 1.0, 1.0)
        glVertex3f(*self.simulator.particles[2].position)
        glVertex3f(*self.simulator.particles[3].position)
        glEnd()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_axes()
        self._draw_particles()
        self._draw_springs()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        glfw.terminate()