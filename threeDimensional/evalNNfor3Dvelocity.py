import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, gaussian_filter
import numpy.ma as ma


import sys
sys.path.append('../python-nn')
from testMuscle import *

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer=3):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(layer-1):
            if k == 0:
                self.hidden.append(nn.Linear(input_size, hidden_size))
            else:
                self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

# Instantiate the model
input_size = 3
hidden_size = 64
output_size = 1
model = MLP(input_size, hidden_size, output_size, 6)

# Load the model's state_dict (parameters)
model.load_state_dict(torch.load('threeDimensional/mlp_model.pth'))

# Set the model to evaluation mode (important for inference)
model.eval()

# Generate eval data
sample_size = 100
lMtilde = np.linspace(.30, 2.00, sample_size)
lTtilde = np.linspace(.985, 1.075, sample_size)
X, Y = np.meshgrid(lMtilde, lTtilde)
act = [1] * X.ravel().shape[0]
X_Input = np.vstack([X.ravel(), Y.ravel(), act]).T
X_Input_tensor = torch.tensor(X_Input, dtype=torch.float32)

with torch.no_grad():  # Disables gradient calculation for faster inference
    predicted_velocity = model(X_Input_tensor).numpy()

# Reshape results into grid shape
Z = predicted_velocity.reshape(X.shape)

# Get the error between GT and prediction, and reshape
# Generate GT
df_GT = pd.read_csv("twoDimensional/velocityData_2D_a=1.csv")
# Organize the prediction such that it has the exact shape as ground truth
DF = pd.DataFrame(np.column_stack([predicted_velocity, df_GT["vMtilde"]]), 
                               columns=['Prediction', 'GT'])
DF.loc[(DF.GT <= -1.75) | (DF.GT >= 1.75), 'Prediction'] = np.nan
Z_error_filtered = (np.asarray(DF['GT']) - np.asarray(DF['Prediction'])).reshape(X.shape)
Z_error = (np.asarray(DF['GT'])).reshape(X.shape) - Z
Z_gt = (np.asarray(DF['GT'])).reshape(X.shape)

# Custom levels for ground truth
gt_levels = np.linspace(-1.75, 1.75, 11)

# Create a fine grid for interpolation (200x200 grid)
X_fine, Y_fine = np.meshgrid(np.linspace(.30, 2.00, sample_size*10), np.linspace(.985, 1.075, sample_size*10))

# Interpolate both the ground truth and the error data onto the fine grid
Z_gt_fine = griddata((X.ravel(), Y.ravel()), Z_gt.ravel(), (X_fine, Y_fine), method='linear')
Z_error_fine = griddata((X.ravel(), Y.ravel()), Z_error.ravel(), (X_fine, Y_fine), method='linear')

# Handle NaNs after interpolation by setting them to 0 or applying a mask
Z_gt_fine = np.nan_to_num(Z_gt_fine, nan=0)
Z_error_fine = np.nan_to_num(Z_error_fine, nan=0)

# Mask areas in the interpolated ground truth data that are outside the desired levels
mask = (Z_gt_fine < -1.75) | (Z_gt_fine > 1.75)

# Apply the same mask to both the ground truth and error data
Z_gt_fine_masked = np.ma.masked_where(mask, Z_gt_fine)
Z_error_fine_masked = np.ma.masked_where(mask, Z_error_fine)

# Create the figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the interpolated and masked ground truth
contour_gt = ax[0].contourf(X_fine, Y_fine, Z_gt_fine_masked, levels=np.linspace(-1.75, 1.75, 11))
fig.colorbar(contour_gt, ax=ax[0], ticks=np.linspace(-1.75, 1.75, 11))
ax[0].set_title('Interpolated Ground Truth')

# Plot the interpolated and masked error
contour_error = ax[1].contourf(X_fine, Y_fine, Z_error_fine_masked)
fig.colorbar(contour_error, ax=ax[1])
ax[1].set_title('Interpolated Error with Same Mask')

"""
# Mask the NaNs and interpolate
points = np.array([X[~np.isnan(Z1)], Y[~np.isnan(Z1)]]).T
values = Z1[~np.isnan(Z1)]
grid_z = griddata(points, values, (X, Y), method='nearest')

z2 = np.asarray(DF['GT']).reshape(X.shape)

# Step 2: Define contour levels for z2 and identify the plotted region
levels_z2 = np.linspace(-1.75, 1.75, 11)  # Define contour levels for z2
plotted_region_mask = np.logical_and(z2 >= min(levels_z2), z2 <= max(levels_z2))

# Step 3: Create a transition (gradient) mask for smoothing the boundary of the unplotted region
# Apply Gaussian filter to soften the boundary between plotted and unplotted areas
distance_to_unplotted = gaussian_filter(plotted_region_mask.astype(float), sigma=5)

# Invert the distance mask to emphasize unplotted regions (lower values near unplotted)
blending_mask = np.clip(1 - distance_to_unplotted, 0, 1)

# Step 4: Apply the blending mask to the first plot
z1_masked = np.copy(grid_z)
z1_masked[blending_mask < 0.05] = np.nan  # Mask out regions where blending is near unplotted


df = pd.read_csv('velocityData_2D_a=1.csv')
Z1 = (df["vMtilde"].to_numpy() - predicted_velocity.squeeze()).reshape(X.shape)

# Plot the true function vs the model prediction
plt.contourf(X, Y, Z, levels=np.linspace(-1.75, 1.75, 11))
plt.colorbar(label='vMtilde', ticks=np.linspace(-1.75, 1.75, 11)) 
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.title('MLP Prediction for act = 1')

# Plot the error
#plt.contourf(X, Y, Z1, levels=np.linspace(-1, 1, 11))
#plt.colorbar(label='vMtilde', ticks=np.linspace(-1, 1, 11)) 
#, levels=np.linspace(-0.4, 1.2, 9)
#plt.contourf(X, Y, Z2, levels=np.linspace(-0.33, 1.03, 8))
plt.contourf(X1, Y1, grid_z)
plt.colorbar(label='Error in vMtilde')
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.title('Error of MLP Prediction for act = 1')
"""

plt.show()