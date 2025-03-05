# Plotter.py
import matplotlib.pyplot as plt
import numpy as np

class SimulationPlotter:
    def __init__(self):
        self.plot_time = 0
        self.time_data = []
        self.xpbd_over_time = []
        self.classic_over_time = []
        self.activation_over_time = []
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(6, 4))
        self.line_x, = self.axs[0].plot([], [], 'r-', label="XPBD Current Length")
        self.line_x1, = self.axs[0].plot([], [], 'b-', label="Classic Current Length")
        self.axs[0].set_xlim(0, 10)
        self.axs[0].set_ylim(-1.0, 1.0)
        self.axs[0].set_title('Muscle Length')
        self.axs[0].set_xlabel('time (s)')
        self.axs[0].set_ylabel('Lengths (m)')
        self.axs[0].legend(loc='upper right')
        
        self.line_y, = self.axs[1].plot([], [], 'y-', label="activation")
        self.axs[1].set_xlim(0, 10)
        self.axs[1].set_ylim(0, 1.0)
        self.axs[1].set_title('Activation')
        self.axs[1].set_xlabel('time (s)')
        self.axs[1].set_ylabel('act')

    def update(self, simulator, dt, activation):
        xpbd_length = np.linalg.norm(simulator.particles[0].position - 
                                   simulator.particles[1].position)
        classic_length = np.linalg.norm(simulator.particles[2].position - 
                                        simulator.particles[3].position)
        
        self.xpbd_over_time.append(xpbd_length)
        self.classic_over_time.append(classic_length)
        self.time_data.append(self.plot_time)
        self.activation_over_time.append(activation)
        self.plot_time += dt

        self.line_x.set_data(self.time_data, self.xpbd_over_time)
        self.line_x1.set_data(self.time_data, self.classic_over_time)
        self.line_y.set_data(self.time_data, self.activation_over_time)
        
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
        
        plt.draw()
        plt.pause(dt)