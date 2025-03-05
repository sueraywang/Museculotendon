import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import sys
import os
sys.path.append(os.path.abspath('ComputeMuscleForce'))
from testMuscle import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
hidden_size = 128
output_size = 1
model = MLP(input_size, hidden_size, output_size, 6)

# Load the saved model weights onto the GPU
model.load_state_dict(torch.load('PlotMuscleVelocity3D/mlp_model.pth', map_location=device))

# Move the model to CPU
model = model.to('cpu')
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
df_GT = pd.read_csv("PlotMuscleVelocity2D/velocityData_2D_a=1.csv")
# Organize the prediction such that it has the exact shape as ground truth
DF = pd.DataFrame(np.column_stack([predicted_velocity, df_GT["vMtilde"]]), 
                               columns=['Prediction', 'GT'])
DF.loc[(DF.GT <= -1.75) | (DF.GT >= 1.75), 'Prediction'] = np.nan
Z_error_uninterpolated = (np.asarray(DF['GT'])).reshape(X.shape) - (np.asarray(DF['Prediction'])).reshape(X.shape)
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

DF1 = pd.DataFrame(np.column_stack([X_fine.ravel(), Y_fine.ravel(), Z_error_fine_masked.ravel()]), 
                               columns=['X', 'Y', 'Z'])
max1 = max(Z_error_fine_masked.ravel())
print(DF1[DF1['Z'] == max1])

DF2 = pd.DataFrame(np.column_stack([X.ravel(), Y.ravel(), Z_error_uninterpolated.ravel()]), 
                               columns=['X', 'Y', 'Z'])
DF2 = DF2.sort_values(by=['Z'], ascending=False)
list1 = DF2['X'].head(5)
list2 = DF2['Y'].head(5)
print(DF2.head(5))
print(max(list1), min(list1))
print(max(list2), min(list2))

# Create the figure and subplots
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# Plot the interpolated and masked ground truth
contour_gt = ax[0].contourf(X_fine, Y_fine, Z_gt_fine_masked, levels=np.linspace(-1.75, 1.75, 11))
fig.colorbar(contour_gt, ax=ax[0], ticks=np.linspace(-1.75, 1.75, 11))
ax[0].set_title('Interpolated Ground Truth')

# Plot the interpolated and masked error
contour_error = ax[1].contourf(X_fine, Y_fine, Z_error_fine_masked)
fig.colorbar(contour_error, ax=ax[1])
ax[1].set_title('Interpolated Error with Same Mask')

# Plot the uninterpolated error
contour_error_no_interpolation = ax[2].contourf(X, Y, Z_error_uninterpolated)
fig.colorbar(contour_error_no_interpolation, ax=ax[2])
ax[2].set_title('Error of MLP Prediction for act = 1 without interpolation')

plt.xlabel('lMtilde')
plt.ylabel('lTtilde')

plt.show()