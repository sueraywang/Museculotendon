# Renderer.py
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import math
import ctypes

class CameraController:
    def __init__(self, width, height, cam_distance=5.0):
        self.target = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        self.camera_distance = cam_distance
        
        # Use right-handed coordinate system (x-right, y-up, z-forward)
        # Initialize camera position to be behind and slightly above the origin
        self.theta = np.pi/2    # Horizontal angle - start behind the scene (-z)
        self.phi = np.pi/2     # Vertical angle - slightly above the scene
        
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # y-up
        self.camera_pos = None
        self.camera_front = None
        
        # For controls
        self.mouse_sensitivity = 0.005
        self.pan_sensitivity = 0.01
        
        # Update camera vectors
        self.spherical_to_cartesian()
    
    def spherical_to_cartesian(self):
        """Convert spherical coordinates to Cartesian (x-right, y-up, z-forward)"""
        # x = r * sin(phi) * cos(theta)
        # y = r * cos(phi)
        # z = r * sin(phi) * sin(theta)
        x = self.camera_distance * np.sin(self.phi) * np.cos(self.theta)
        y = self.camera_distance * np.cos(self.phi)
        z = self.camera_distance * np.sin(self.phi) * np.sin(self.theta)
        
        self.camera_pos = self.target + np.array([x, y, z], dtype=np.float32)
        self.camera_front = (self.target - self.camera_pos) / np.linalg.norm(self.target - self.camera_pos)
    
    def get_view_matrix(self):
        """Returns the view matrix for the current camera position"""
        return pyrr.matrix44.create_look_at(
            self.camera_pos, 
            self.target,
            self.camera_up
        )
    
    def process_mouse_movement(self, x_offset, y_offset, orbit_active=False, pan_active=False):
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity
        
        if orbit_active:
            self.theta -= x_offset
            self.phi -= y_offset
            self.phi = np.clip(self.phi, 0.1, np.pi - 0.1)
            self.spherical_to_cartesian()
        elif pan_active:
            # Pan in the camera's local xy plane
            right = np.cross(self.camera_front, self.camera_up)
            right = right / np.linalg.norm(right)
            up = self.camera_up / np.linalg.norm(self.camera_up)
            
            pan_x = -x_offset * self.pan_sensitivity * self.camera_distance
            pan_y = y_offset * self.pan_sensitivity * self.camera_distance
            
            self.target += right * pan_x + up * pan_y
            self.spherical_to_cartesian()
    
    def process_mouse_scroll(self, y_offset):
        self.camera_distance -= y_offset * 0.1
        self.camera_distance = max(0.5, min(10.0, self.camera_distance))
        self.spherical_to_cartesian()

class Sphere:
    def __init__(self, radius=0.05, slices=16, stacks=16):
        self.radius = radius
        self.slices = slices
        self.stacks = stacks
        
        vertices = []
        indices = []
        
        # Create sphere vertices in spherical coordinates
        # Consistent with right-handed system (x-right, y-up, z-forward)
        for i in range(stacks + 1):
            v = i / stacks
            phi = v * math.pi  # 0 to π (from top to bottom)
            
            for j in range(slices + 1):
                u = j / slices
                theta = u * 2 * math.pi  # 0 to 2π (around the sphere)
                
                # Spherical to Cartesian conversion for right-handed system
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                # Normal is just the normalized position for a sphere
                nx = math.sin(phi) * math.cos(theta)
                ny = math.cos(phi)
                nz = math.sin(phi) * math.sin(theta)
                
                vertices.extend([x, y, z, nx, ny, nz])
        
        # Create indices for sphere triangles
        for i in range(stacks):
            for j in range(slices):
                first = i * (slices + 1) + j
                second = first + slices + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        
        self.setup_buffers()
    
    def setup_buffers(self):
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create VBO
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

class Line:
    def __init__(self, start=[0, 0, 0], end=[1, 0, 0]):
        self.vertices = np.array([
            start[0], start[1], start[2],
            end[0], end[1], end[2]
        ], dtype=np.float32)
        
        self.setup_buffers()
    
    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glBindVertexArray(0)
    
    def update(self, start, end):
        self.vertices = np.array([
            start[0], start[1], start[2],
            end[0], end[1], end[2]
        ], dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
    
    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_LINES, 0, 2)  # Only draw 2 vertices
        glBindVertexArray(0)

class Renderer:
    def __init__(self, simulator, width=800, height=600, cam_distance=5.0):
        self.simulator = simulator
        self.width = width
        self.height = height
        
        self._init_glfw()
        self._init_camera(cam_distance)
        self._setup_shaders()
        self._setup_geometry()
        self._setup_lighting()
    
    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Use core profile for modern OpenGL
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        self.window = glfw.create_window(self.width, self.height, "Spring Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        
        # Mouse state
        self.last_x = self.width / 2
        self.last_y = self.height / 2
        self.first_mouse = True
        self.right_mouse_pressed = False
        self.middle_mouse_pressed = False
    
    def _init_camera(self, cam_distance):
        # Initialize with a specific viewing angle for x-right, y-up, z-forward
        self.camera = CameraController(self.width, self.height, cam_distance=cam_distance)
    
    def _framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.width, self.height = width, height
    
    def _mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return
        
        x_offset = self.last_x - xpos
        y_offset = ypos - self.last_y  # Reversed: y ranges bottom to top
        self.last_x = xpos
        self.last_y = ypos
        
        self.camera.process_mouse_movement(x_offset, y_offset, 
                                         orbit_active=self.right_mouse_pressed,
                                         pan_active=self.middle_mouse_pressed)
    
    def _scroll_callback(self, window, xoffset, yoffset):
        self.camera.process_mouse_scroll(yoffset)
    
    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.right_mouse_pressed = action == glfw.PRESS
            if action == glfw.PRESS:
                self.first_mouse = True
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.middle_mouse_pressed = action == glfw.PRESS
            if action == glfw.PRESS:
                self.first_mouse = True
    
    def _setup_shaders(self):
        # Vertex shader for objects (spheres) - Right-handed coordinate system
        sphere_vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        
        out vec3 FragPos;
        out vec3 Normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        # Fragment shader for objects
        sphere_fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        
        void main()
        {
            // Ambient
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;
            
            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
        """
        
        # Vertex shader for lines
        line_vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
        """
        
        # Fragment shader for lines
        line_fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        uniform vec3 lineColor;
        
        void main()
        {
            FragColor = vec4(lineColor, 1.0);
        }
        """
        
        # Vertex shader for ground plane
        ground_vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        
        out vec3 FragPos;
        out vec3 Normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = aNormal;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        # Fragment shader for ground plane
        ground_fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 gridColor1;
        uniform vec3 gridColor2;
        uniform float gridSize;
        
        void main()
        {
            // Create grid effect
            float x = FragPos.x;
            float z = FragPos.z;
            
            // Determine grid cell - x-z plane for y-up
            bool isGridX = mod(floor(x / gridSize), 2.0) == 0.0;
            bool isGridZ = mod(floor(z / gridSize), 2.0) == 0.0;
            
            // Choose color based on checkerboard pattern
            vec3 baseColor = (isGridX == isGridZ) ? gridColor1 : gridColor2;
            
            // Lighting calculations
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(1.0);
            
            // Final color
            vec3 result = baseColor * (0.3 + 0.7 * diffuse);
            FragColor = vec4(result, 1.0);
        }
        """
        
        # Compile shaders
        sphere_vertex = shaders.compileShader(sphere_vertex_shader, GL_VERTEX_SHADER)
        sphere_fragment = shaders.compileShader(sphere_fragment_shader, GL_FRAGMENT_SHADER)
        line_vertex = shaders.compileShader(line_vertex_shader, GL_VERTEX_SHADER)
        line_fragment = shaders.compileShader(line_fragment_shader, GL_FRAGMENT_SHADER)
        ground_vertex = shaders.compileShader(ground_vertex_shader, GL_VERTEX_SHADER)
        ground_fragment = shaders.compileShader(ground_fragment_shader, GL_FRAGMENT_SHADER)
        
        # Link shaders
        self.sphere_shader = shaders.compileProgram(sphere_vertex, sphere_fragment)
        self.line_shader = shaders.compileProgram(line_vertex, line_fragment)
        self.ground_shader = shaders.compileProgram(ground_vertex, ground_fragment)
    
    def _setup_geometry(self):
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Create sphere for particles
        self.sphere = Sphere(radius=0.05)
        
        # Create lines for springs
        self.spring_lines = [
            Line(),  # XPBD spring
            Line()   # Newtonian spring
        ]
        
        # Create grid for ground plane
        self._setup_ground_plane()
    
    def _setup_ground_plane(self):
        # Create a simple grid as the ground plane in the xz-plane (y=0)
        size = 10.0
        vertices = np.array([
            -size, 0.0, -size,  0.0, 1.0, 0.0,  # Normal points up (y)
             size, 0.0, -size,  0.0, 1.0, 0.0,
             size, 0.0,  size,  0.0, 1.0, 0.0,
            -size, 0.0,  size,  0.0, 1.0, 0.0
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)
        
        # Create and bind VAO
        self.ground_vao = glGenVertexArrays(1)
        glBindVertexArray(self.ground_vao)
        
        # Create and bind VBO
        self.ground_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ground_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create and bind EBO
        self.ground_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ground_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def _setup_lighting(self):
        # Light position in the scene
        self.light_pos = np.array([2.0, 5.0, 2.0], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    def _update_spring_lines(self):
        # Update XPBD spring
        self.spring_lines[0].update(
            self.simulator.particles[0].position,
            self.simulator.particles[1].position
        )
        
        # Update Newtonian spring
        self.spring_lines[1].update(
            self.simulator.particles[2].position,
            self.simulator.particles[3].position
        )
    
    def render(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update spring lines
        self._update_spring_lines()
        
        # Get view/projection matrices
        view = self.camera.get_view_matrix()
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            45.0, self.width / self.height, 0.1, 100.0
        )
        
        # Draw ground plane
        self._render_ground(view, projection)
        
        # Draw springs
        self._render_springs(view, projection)
        
        # Draw particles
        self._render_particles(view, projection)
        
        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def _render_ground(self, view, projection):
        glUseProgram(self.ground_shader)
        
        # Set matrices
        model_loc = glGetUniformLocation(self.ground_shader, "model")
        view_loc = glGetUniformLocation(self.ground_shader, "view")
        proj_loc = glGetUniformLocation(self.ground_shader, "projection")
        
        model = pyrr.matrix44.create_identity()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        
        # Set lighting and grid properties
        light_pos_loc = glGetUniformLocation(self.ground_shader, "lightPos")
        view_pos_loc = glGetUniformLocation(self.ground_shader, "viewPos")
        grid_color1_loc = glGetUniformLocation(self.ground_shader, "gridColor1")
        grid_color2_loc = glGetUniformLocation(self.ground_shader, "gridColor2")
        grid_size_loc = glGetUniformLocation(self.ground_shader, "gridSize")
        
        glUniform3fv(light_pos_loc, 1, self.light_pos)
        glUniform3fv(view_pos_loc, 1, self.camera.camera_pos)
        glUniform3f(grid_color1_loc, 0.9, 0.9, 0.9)  # Light gray
        glUniform3f(grid_color2_loc, 0.3, 0.3, 0.3)  # Dark gray
        glUniform1f(grid_size_loc, 0.5)  # Grid cell size
        
        # Draw ground
        glBindVertexArray(self.ground_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
    
    def _render_springs(self, view, projection):
        glUseProgram(self.line_shader)
        
        model_loc = glGetUniformLocation(self.line_shader, "model")
        view_loc = glGetUniformLocation(self.line_shader, "view")
        proj_loc = glGetUniformLocation(self.line_shader, "projection")
        color_loc = glGetUniformLocation(self.line_shader, "lineColor")
        
        model = pyrr.matrix44.create_identity()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        
        # Draw XPBD spring (red)
        glUniform3f(color_loc, 1.0, 0.0, 0.0)
        self.spring_lines[0].draw()
        
        # Draw Newtonian spring (cyan)
        glUniform3f(color_loc, 0.0, 1.0, 1.0)
        self.spring_lines[1].draw()
    
    def _render_particles(self, view, projection):
        glUseProgram(self.sphere_shader)
        
        model_loc = glGetUniformLocation(self.sphere_shader, "model")
        view_loc = glGetUniformLocation(self.sphere_shader, "view")
        proj_loc = glGetUniformLocation(self.sphere_shader, "projection")
        light_pos_loc = glGetUniformLocation(self.sphere_shader, "lightPos")
        view_pos_loc = glGetUniformLocation(self.sphere_shader, "viewPos")
        light_color_loc = glGetUniformLocation(self.sphere_shader, "lightColor")
        obj_color_loc = glGetUniformLocation(self.sphere_shader, "objectColor")
        
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniform3fv(light_pos_loc, 1, self.light_pos)
        glUniform3fv(view_pos_loc, 1, self.camera.camera_pos)
        glUniform3fv(light_color_loc, 1, self.light_color)
        
        for i, particle in enumerate(self.simulator.particles):
            if i > 4:
                break
            # Set color based on particle type
            if particle.fixed:  # Fixed particles
                glUniform3f(obj_color_loc, 0.0, 1.0, 0.0)  # Red
            else:  # Moving particles
                if particle.xpbd:
                    glUniform3f(obj_color_loc, 1.0, 0.0, 0.0)  # Green for XPBD
                else:
                    glUniform3f(obj_color_loc, 0.0, 0.8, 1.0)  # Blue-cyan for classic
            
            # Set model matrix (translation to particle position)
            model = pyrr.matrix44.create_identity()
            model = pyrr.matrix44.multiply(
                model,
                pyrr.matrix44.create_from_translation(particle.position)
            )
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            
            # Draw sphere
            self.sphere.draw()
    
    def should_close(self):
        return glfw.window_should_close(self.window)
    
    def process_input(self):
        # Handle escape key to close window
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
    
    def cleanup(self):
        # Delete sphere resources
        glDeleteVertexArrays(1, [self.sphere.VAO])
        glDeleteBuffers(1, [self.sphere.VBO])
        glDeleteBuffers(1, [self.sphere.EBO])
        
        # Delete line resources
        for line in self.spring_lines:
            glDeleteVertexArrays(1, [line.VAO])
            glDeleteBuffers(1, [line.VBO])
        
        # Delete ground resources
        glDeleteVertexArrays(1, [self.ground_vao])
        glDeleteBuffers(1, [self.ground_vbo])
        glDeleteBuffers(1, [self.ground_ebo])
        
        # Delete shader programs
        glDeleteProgram(self.sphere_shader)
        glDeleteProgram(self.line_shader)
        glDeleteProgram(self.ground_shader)
        
        # Terminate GLFW
        glfw.terminate()