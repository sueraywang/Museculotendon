import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

df = pd.read_csv('PlotMuscleVelocity3D/random3DVelocityData.csv')

tolerance = 0.05  # Adjust this as needed
df_slice = df[df["act"] == 1]

# Create grid values for X and Y
df_slice = df_slice.sort_values(by=['lMtilde', 'lTtilde'])
xi = df_slice["lMtilde"].unique()
yi = df_slice["lTtilde"].unique()
# print(df_slice)
# print(yi)

X, Y = np.meshgrid(xi, yi)

# Interpolate v values over the grid
Z = griddata((df_slice["lMtilde"], df_slice["lTtilde"]),
                 df_slice["vMtilde"], (X, Y), method='cubic')

# Create the contour plot
plt.figure()
plt.tricontourf(df_slice["lMtilde"], df_slice["lTtilde"], df_slice["vMtilde"])
#plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.5, 11))
# Add colorbar for reference
plt.colorbar(label='vMtilde')
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.title(f'Contour Plot for act is 1')
plt.show()
