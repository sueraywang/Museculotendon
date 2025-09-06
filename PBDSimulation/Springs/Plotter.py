# Plotter.py
import matplotlib.pyplot as plt
import numpy as np

class SimulationPlotter:
    def __init__(self):
        self.plot_time = 0
        self.time_data = []
        self.xpbd_over_time = []
        self.classic_over_time = []
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        
        self.fig, self.axs = plt.subplots(1, 1, figsize=(12, 8))  # self.axs is a single AxesSubplot
        self.line_x, = self.axs.plot([], [], 'r-', label="XPBD Current Length")
        self.line_x1, = self.axs.plot([], [], 'b-', label="Classic Current Length")
        self.axs.set_xlim(0, 10)
        self.axs.set_ylim(-1.0, 1.0)
        self.axs.set_title('Spring Length')
        self.axs.set_xlabel('time (s)')
        self.axs.set_ylabel('Length (m)')
        self.axs.legend(loc='upper right')

        
    def update(self, simulator, dt):
        xpbd_length = np.linalg.norm(simulator.particles[0].position - 
                                   simulator.particles[1].position)
        classic_length = np.linalg.norm(simulator.particles[2].position - 
                                        simulator.particles[3].position)
        
        self.xpbd_over_time.append(xpbd_length)
        self.classic_over_time.append(classic_length)
        self.time_data.append(self.plot_time)
        self.plot_time += dt

        self.line_x.set_data(self.time_data, self.xpbd_over_time)
        self.line_x1.set_data(self.time_data, self.classic_over_time)
        
        # Adjust x-axis dynamically
        self.axs.set_xlim(0, max(self.time_data))  # Adjust time window

        # Adjust y-axis dynamically
        if self.xpbd_over_time:  # Ensure data is not empty before accessing min/max
            self.axs.set_ylim(
                0.0,
                2.0
            )
        else:
            self.axs.set_ylim(-1.0, 1.0)  # Default range if data is empty
        
        plt.draw()
        plt.pause(dt)