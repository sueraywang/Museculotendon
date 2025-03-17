# Plotter.py
import matplotlib.pyplot as plt

class SimulationPlotter:
    def __init__(self, muscle=False):
        self.plot_time = 0
        self.muscle = muscle
        self.time_data = []
        self.xpbd_over_time = []
        self.classic_over_time = []
        if self.muscle:
            self.activation_over_time = []
            self.pennation_over_time = []
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        if self.muscle:
            self.fig, self.axs = plt.subplots(3, 1, figsize=(6, 10))
            self.line_x, = self.axs[0].plot([], [], 'r-', label="XPBD Current Position")
            self.line_x1, = self.axs[0].plot([], [], 'b-', label="Classic Current Position")
            self.axs[0].set_xlim(0, 10)
            self.axs[0].set_ylim(-1.0, 1.0)
            self.axs[0].set_title('Moving Particle Position')
            self.axs[0].set_xlabel('time (s)')
            self.axs[0].set_ylabel('Position (m)')
            self.axs[0].legend(loc='upper right')
            
            self.line_y, = self.axs[1].plot([], [], 'y-', label="activation")
            self.axs[1].set_xlim(0, 10)
            self.axs[1].set_ylim(0, 1.0)
            self.axs[1].set_title('Activation')
            self.axs[1].set_xlabel('time (s)')
            self.axs[1].set_ylabel('act')
            
            self.line_z, = self.axs[2].plot([], [], 'g-', label="pennation")
            self.axs[2].set_xlim(0, 10)
            self.axs[2].set_ylim(0, 1.0)
            self.axs[2].set_title('Pennation')
            self.axs[2].set_xlabel('time (s)')
            self.axs[2].set_ylabel('penn')
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(6, 4))  # self.axs is a single AxesSubplot
            self.line_x, = self.axs.plot([], [], 'r-', label="XPBD Current Position")
            self.line_x1, = self.axs.plot([], [], 'b-', label="Classic Current Position")
            self.axs.set_xlim(0, 10)
            self.axs.set_ylim(-1.0, 1.0)
            self.axs.set_title('Moving Particle Position')
            self.axs.set_xlabel('time (s)')
            self.axs.set_ylabel('Position (m)')
            self.axs.legend(loc='upper right')

        
    def update(self, simulator, dt, activation, pennation):
        #"""
        xpbd_length = simulator.particles[1].position[1]
        classic_length = simulator.particles[3].position[1]
        """
        xpbd_length = np.linalg.norm(simulator.particles[0].position - 
                                   simulator.particles[1].position)
        classic_length = np.linalg.norm(simulator.particles[2].position - 
                                        simulator.particles[3].position)
        """
        
        self.xpbd_over_time.append(xpbd_length)
        self.classic_over_time.append(classic_length)
        self.time_data.append(self.plot_time)
        if self.muscle:
            self.activation_over_time.append(activation)
            self.pennation_over_time.append(pennation)
        self.plot_time += dt

        self.line_x.set_data(self.time_data, self.xpbd_over_time)
        self.line_x1.set_data(self.time_data, self.classic_over_time)
        if self.muscle:
            self.line_y.set_data(self.time_data, self.activation_over_time)
            self.line_z.set_data(self.time_data, self.pennation_over_time)
        
        if self.muscle:
            # Adjust x-axis dynamically for each subplot
            for ax in self.axs:
                ax.set_xlim(0, max(self.time_data))  # Adjust time window
            # Adjust y-axis dynamically for each subplot
            self.axs[0].set_ylim(
                min(min(self.xpbd_over_time), min(self.classic_over_time)),
                max(max(self.xpbd_over_time), max(self.classic_over_time)) 
                if self.xpbd_over_time else 1.0
            )
            self.axs[1].set_ylim(min(self.activation_over_time), max(self.activation_over_time))
            self.axs[2].set_ylim(min(self.pennation_over_time), max(self.pennation_over_time))
        else:
            # Adjust x-axis dynamically
            self.axs.set_xlim(0, max(self.time_data))  # Adjust time window

            # Adjust y-axis dynamically
            if self.xpbd_over_time:  # Ensure data is not empty before accessing min/max
                self.axs.set_ylim(
                    min(min(self.xpbd_over_time), min(self.classic_over_time)),
                    max(max(self.xpbd_over_time), max(self.classic_over_time))
                )
            else:
                self.axs.set_ylim(-1.0, 1.0)  # Default range if data is empty
        
        plt.draw()
        plt.pause(dt)