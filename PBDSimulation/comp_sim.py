import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Results/precomputed_model_100steps.csv')

time_points = df['time']
xpbd_lengths = df['xpbd_length']
classical_lengths = df['classic_length']

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- First plot: positions ---
axs[0].plot(time_points, classical_lengths, 'b-', label='Classical Simulation Position')
axs[0].plot(time_points, xpbd_lengths, 'r--', label='XPBD Simulation Position')
axs[0].set_ylabel('Position (m)')
axs[0].set_title('Comparison of Positions Between Simulations')
axs[0].legend()
axs[0].grid(True)

# --- Second plot: absolute difference ---
pos_diff = np.abs(classical_lengths - xpbd_lengths)
axs[1].plot(time_points, pos_diff, 'g-')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Absolute Difference (m)')
axs[1].set_title('Absolute Difference in Positions')
axs[1].grid(True)

plt.tight_layout()
plt.savefig("Results/precomputed_model_100steps")
plt.show()
