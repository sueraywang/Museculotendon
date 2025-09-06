# Plotter.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SimulationPlotter:
    def __init__(self, MT=False):
        self.MT = MT
        self.plot_time = 0
        self.time_data = []
        self.vel_data = []
        self.xpbd_over_time = []
        self.classic_over_time = []
        self.activation_over_time = []
        self.pennation_over_time = []
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(6, 12))
        self.line_x, = self.axs[0].plot([], [], 'r-', label="XPBD Current Length")
        self.line_x1, = self.axs[0].plot([], [], 'b-', label="Classic Current Length")
        self.axs[0].set_xlim(0, 10)
        self.axs[0].set_ylim(-1.0, 1.0)
        self.axs[0].set_title('Muscle Length')
        self.axs[0].set_xlabel('time (s)')
        self.axs[0].set_ylabel('Position (m)')
        self.axs[0].legend(loc='lower right')
        
        self.line_y, = self.axs[1].plot([], [], 'y-', label="activation")
        self.axs[1].set_xlim(0, 10)
        self.axs[1].set_ylim(0, 1.1)
        self.axs[1].set_title('Activation')
        self.axs[1].set_xlabel('time (s)')
        self.axs[1].set_ylabel('act')
        
        self.line_z, = self.axs[2].plot([], [], 'g-', label="pennation")
        self.axs[2].set_xlim(0, 10)
        self.axs[2].set_ylim(0, 1.0)
        self.axs[2].set_title('Pennation')
        self.axs[2].set_xlabel('time (s)')
        self.axs[2].set_ylabel('penn')

        
    def update(self, simulator, dt, activation, pennation):
        if self.MT:
            xpbd_length = simulator.xpbd_l_M
            classic_length = simulator.classic_l_M
        else:
            xpbd_length = np.linalg.norm(simulator.particles[0].position - 
                                    simulator.particles[1].position)
            classic_length = np.linalg.norm(simulator.particles[2].position - 
                                            simulator.particles[3].position)
        
        self.xpbd_over_time.append(xpbd_length)
        self.classic_over_time.append(classic_length)
        self.time_data.append(self.plot_time)
        self.vel_data.append(np.linalg.norm(simulator.particles[3].velocity))
        self.activation_over_time.append(activation)
        self.pennation_over_time.append(simulator.penn)
        self.plot_time += dt

        self.line_x.set_data(self.time_data, self.xpbd_over_time)
        self.line_x1.set_data(self.time_data, self.classic_over_time)
        self.line_y.set_data(self.time_data, self.activation_over_time)
        self.line_z.set_data(self.time_data, self.pennation_over_time)
        
        # Adjust x-axis dynamically for each subplot
        for ax in self.axs:
            ax.set_xlim(0.0, max(1,0, max(self.time_data)))  # Adjust time window
        # Adjust y-axis dynamically for each subplot
        self.axs[0].set_ylim(
            0.0, max(0.15, max(self.xpbd_over_time), max(self.classic_over_time)))
        self.axs[1].set_ylim(-0.1, 1.1)
        self.axs[2].set_ylim(0.0, max(self.pennation_over_time))
        
        plt.draw()
        plt.pause(dt)
        
    def export_data(self, filename="xpbd_simulation_data.csv"):        
        # Create a dictionary with all the data
        data = {
            'time': self.time_data,
            'xpbd_length': self.xpbd_over_time,
            'classic_length': self.classic_over_time
        }
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Simulation data exported to {filename}")
        
        return df