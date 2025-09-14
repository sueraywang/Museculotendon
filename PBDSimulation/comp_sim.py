import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
df = pd.read_csv('Results/damp_64by3.csv')

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
plt.savefig("Results/damp_64by3")
plt.show()
"""

df1 = pd.read_csv('Results/act_len_64by3.csv')
df2 = pd.read_csv('Results/damp_64by3.csv')
df3 = pd.read_csv('Results/fullForce_64by3.csv')

time_points = df1['time']
xpbd_lengths1 = df1['xpbd_length']
classical_lengths1 = df1['classic_length']
error1 = np.abs(classical_lengths1 - xpbd_lengths1)

xpbd_lengths2 = df2['xpbd_length']
classical_lengths2 = df2['classic_length']
error2 = np.abs(classical_lengths2 - xpbd_lengths2)

xpbd_lengths3 = df3['xpbd_length']
classical_lengths3 = df3['classic_length']
error3 = np.abs(classical_lengths3 - xpbd_lengths3)

plt.plot(time_points, error1, label="no_damp_error")
plt.plot(time_points, error2, label="no_penn_error")
plt.plot(time_points, error3, label="full_force_error")

plt.xlabel("Time")
plt.ylabel("Error (|classic - xpbd|)")
plt.title("Error Curves")
plt.legend()
plt.grid(True)
plt.ylim(0, 0.0005)
plt.savefig("Results/error_comparisons_MT")

plt.show()
#"""