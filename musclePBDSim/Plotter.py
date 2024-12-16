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
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.line_x, = self.ax.plot([], [], 'r-', label="XPBD Current Length")
        self.line_x1, = self.ax.plot([], [], 'b-', label="Classic Current Length")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_title('Cubic Spring Length')
        self.ax.set_xlabel('time (s)')
        self.ax.set_ylabel('Lengths (m)')
        self.ax.legend(loc='upper right')

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
        
        self.ax.set_xlim(0, max(self.time_data) if self.time_data else 10)
        self.ax.set_ylim(
            min(min(self.xpbd_over_time), min(self.classic_over_time)),
            max(max(self.xpbd_over_time), max(self.classic_over_time)) 
            if self.xpbd_over_time else 1.0
        )
        
        plt.draw()
        plt.pause(dt)