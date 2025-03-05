import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import rc
from scipy.interpolate import griddata

# View angles
ELEV = 30
AZIM = 100
FIG_WID = 12
FIG_HT = 10
LABEL_SIZE = 16
TITLE_SIZE = 20

# Set up LaTeX rendering
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the new dataset
data = np.load('TrainingData/lM_act_force.npz')
lMtilde = data['L']      # M×N array
activation = data['A']   # M×N array
muscle_force = data['F'] # M×N array

# Flatten all arrays directly
lMtilde_flat = lMtilde.flatten()
activation_flat = activation.flatten()
muscle_force_flat = muscle_force.flatten()

# Create input array with shape (num_samples, 2)
inputs = np.column_stack((lMtilde_flat, activation_flat))
outputs = muscle_force_flat.reshape(-1, 1)

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
checkpoint = torch.load('TrainedModels/muscle_length_and_act_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Add visualization code
print("Creating visualizations...")

# Create a grid of test points for visualization
lm_min, lm_max = 0.5, np.max(inputs[:, 0])
act_min, act_max = np.min(inputs[:, 1]), np.max(inputs[:, 1])
lm_range = np.linspace(lm_min, lm_max, 100)
act_range = np.linspace(act_min, act_max, 100)
lm_grid, act_grid = np.meshgrid(lm_range, act_range)

# Generate test data points from the grid
test_inputs = np.zeros((lm_grid.size, 2))
test_inputs[:, 0] = lm_grid.flatten()
test_inputs[:, 1] = act_grid.flatten()
test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)

# For predictions without gradients
model.eval()
with torch.no_grad():
    test_C = model(test_inputs_tensor).cpu().numpy()

# Compute gradients for visualization (dC/dlM)
# Using smaller batches to avoid memory issues
batch_size = 1000
num_batches = (test_inputs.shape[0] + batch_size - 1) // batch_size
grads = np.zeros((test_inputs.shape[0], 1))

for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min((b + 1) * batch_size, test_inputs.shape[0])
    
    batch_inputs = test_inputs_tensor[start_idx:end_idx].clone().detach().requires_grad_(True)
    batch_outputs = model(batch_inputs)
    
    batch_grad = torch.autograd.grad(
        outputs=batch_outputs, 
        inputs=batch_inputs,
        grad_outputs=torch.ones_like(batch_outputs),
        create_graph=False
    )[0].cpu().numpy()[:, 0:1]  # Only take gradient wrt lMtilde
    
    grads[start_idx:end_idx] = batch_grad

# Calculate C*dC/dlM (the predicted force)
computed_force = test_C.flatten() * grads.flatten()

# Reshape for plotting
C_reshaped = test_C.reshape(act_grid.shape)
grads_reshaped = grads.reshape(act_grid.shape)
computed_force_reshaped = computed_force.reshape(act_grid.shape)

# Create and save individual plots

# Plot 1: Learned C(x) Function as a 3D surface
fig = plt.figure(figsize=(FIG_WID, FIG_HT))
ax1 = plt.subplot(111, projection='3d')
surf1 = ax1.plot_surface(lm_grid, act_grid, C_reshaped, cmap='viridis', alpha=0.8)
ax1.view_init(elev=ELEV, azim=AZIM)
ax1.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax1.set_title(r'Learned Constraint', fontsize=TITLE_SIZE)
cbar1 = plt.colorbar(surf1, ax=ax1)
cbar1.set_label(label=r'$C(x)$', fontsize=LABEL_SIZE)
plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/01_Cx_function.png', dpi=300, bbox_inches='tight')

# Plot 2: Gradient dC/dlM as a 3D surface
plt.figure(figsize=(FIG_WID, FIG_HT))
ax2 = plt.subplot(111, projection='3d')
surf2 = ax2.plot_surface(lm_grid, act_grid, grads_reshaped, cmap='plasma', alpha=0.8)
ax2.view_init(elev=ELEV, azim=AZIM)
ax2.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax2.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax2.set_title(r'Learned Gradient', fontsize=TITLE_SIZE)
cbar2 = plt.colorbar(surf2, ax=ax2)
cbar2.set_label(label=r'$\frac{\partial C}{\partial \tilde{l}_M}$', fontsize=LABEL_SIZE)
plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/02_dCdlM_gradient.png', dpi=300, bbox_inches='tight')

# Plot 3: Computed Force (C*dC/dlM) as a 3D surface
plt.figure(figsize=(FIG_WID, FIG_HT))
ax3 = plt.subplot(111, projection='3d')
surf3 = ax3.plot_surface(lm_grid, act_grid, -computed_force_reshaped, cmap='coolwarm', alpha=0.8)
ax3.view_init(elev=ELEV, azim=AZIM)
ax3.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax3.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax3.set_title(r'Predicted Muscle Force', fontsize=TITLE_SIZE)
cbar3 = plt.colorbar(surf3, ax=ax3)
cbar3.set_label(label=r'$-C \cdot \frac{\partial C}{\partial \tilde{l}_M}$', fontsize=LABEL_SIZE)
plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/03_computed_force_3d.png', dpi=300, bbox_inches='tight')

# Plot 4: 2D contour plot of the computed force
plt.figure(figsize=(FIG_WID, FIG_HT))
ax4 = plt.subplot(111)
contour = ax4.contourf(lm_grid, act_grid, -computed_force_reshaped, 20, cmap='coolwarm')
ax4.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax4.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax4.set_title(r'Predicted Muscle Force Contour Map', fontsize=TITLE_SIZE)
cbar4 = plt.colorbar(contour, ax=ax4)
cbar4.set_label(label=r'$-C \cdot \frac{\partial C}{\partial \tilde{l}_M}$', fontsize=LABEL_SIZE)
plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/04_force_contour.png', dpi=300, bbox_inches='tight')

# Plot 5: Error Comparison between Ground Truth and Prediction
original_inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# Get model predictions for the original data points
model.eval()
with torch.no_grad():
    predicted_force = model(original_inputs_tensor).cpu().numpy().flatten()

# Calculate gradients for the original data points
original_grads = np.zeros((inputs.shape[0], 1))
batch_size = 1000
num_batches = (inputs.shape[0] + batch_size - 1) // batch_size

for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min((b + 1) * batch_size, inputs.shape[0])
    
    batch_inputs = original_inputs_tensor[start_idx:end_idx].clone().detach().requires_grad_(True)
    batch_outputs = model(batch_inputs)
    
    batch_grad = torch.autograd.grad(
        outputs=batch_outputs, 
        inputs=batch_inputs,
        grad_outputs=torch.ones_like(batch_outputs),
        create_graph=False
    )[0].cpu().numpy()[:, 0:1]  # Only take gradient wrt lMtilde
    
    original_grads[start_idx:end_idx] = batch_grad

# Calculate the predicted force using C*dC/dlM formula
predicted_force_with_grad = -predicted_force * original_grads.flatten()

# Calculate absolute error between predicted and ground truth forces
abs_error = np.abs(predicted_force_with_grad - muscle_force_flat)

# Create a figure for the absolute error contour plot - using the same style as other plots
plt.figure(figsize=(FIG_WID, FIG_HT))

# Define grid for interpolation
lm_min, lm_max = 0.5, np.max(inputs[:, 0])
act_min, act_max = np.min(inputs[:, 1]), np.max(inputs[:, 1])
lm_grid_points = np.linspace(lm_min, lm_max, 100)
act_grid_points = np.linspace(act_min, act_max, 100)
lm_grid, act_grid = np.meshgrid(lm_grid_points, act_grid_points)
grid_z = griddata((inputs[:, 0], inputs[:, 1]), abs_error, 
                  (lm_grid, act_grid), method='linear')

# Create contour plot
contour = plt.contourf(lm_grid, act_grid, grid_z, 20, cmap='viridis')
cbar5 = plt.colorbar(contour, shrink=0.5, aspect=5)
cbar5.set_label(label=r'$|F_{\text{pred}} - F_{\text{GT}}|$', fontsize=LABEL_SIZE)

# Use the same LaTeX formatting as in your other plots
plt.xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
plt.ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
plt.title(r'Absolute Error Between Predicted and Ground Truth Forces', fontsize=TITLE_SIZE)
plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/05_absolute_error_contour.png', dpi=300, bbox_inches='tight')

# Add a new combined plot (plots 1, 2, 3, and 5 in one figure)
plt.figure(figsize=(2*FIG_WID, 2*FIG_HT))

# Plot 1: Learned C(x) Function (top-left)
ax1 = plt.subplot(2, 2, 1, projection='3d')
surf1 = ax1.plot_surface(lm_grid, act_grid, C_reshaped, cmap='viridis', alpha=0.8)
ax1.view_init(elev=ELEV, azim=AZIM)
ax1.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax1.set_title(r'(a) Learned Constraint', fontsize=TITLE_SIZE)
cbar1 = plt.colorbar(surf1, ax=ax1, shrink=0.7)
cbar1.set_label(label=r'$C(x)$', fontsize=LABEL_SIZE)

# Plot 2: Gradient dC/dlM (top-right)
ax2 = plt.subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(lm_grid, act_grid, grads_reshaped, cmap='plasma', alpha=0.8)
ax2.view_init(elev=ELEV, azim=AZIM)
ax2.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax2.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax2.set_title(r'(b) Learned Gradient', fontsize=TITLE_SIZE)
cbar2 = plt.colorbar(surf2, ax=ax2, shrink=0.7)
cbar2.set_label(label=r'$\frac{\partial C}{\partial \tilde{l}_M}$', fontsize=LABEL_SIZE)

# Plot 3: Computed Force (C*dC/dlM) (bottom-left)
ax3 = plt.subplot(2, 2, 3, projection='3d')
surf3 = ax3.plot_surface(lm_grid, act_grid, -computed_force_reshaped, cmap='coolwarm', alpha=0.8)
ax3.view_init(elev=ELEV, azim=AZIM)
ax3.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax3.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax3.set_title(r'(c) Predicted Muscle Force', fontsize=TITLE_SIZE)
cbar3 = plt.colorbar(surf3, ax=ax3, shrink=0.7)
cbar3.set_label(label=r'$-C \cdot \frac{\partial C}{\partial \tilde{l}_M}$', fontsize=LABEL_SIZE)

# Plot 5: Error Comparison (bottom-right)
ax5 = plt.subplot(2, 2, 4)
contour = ax5.contourf(lm_grid, act_grid, grid_z, 20, cmap='viridis')
ax5.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=LABEL_SIZE)
ax5.set_ylabel(r'Activation ($a$)', fontsize=LABEL_SIZE)
ax5.set_title(r'(d) Absolute Error', fontsize=TITLE_SIZE)
cbar5 = plt.colorbar(contour, ax=ax5, shrink=0.7)
cbar5.set_label(label=r'$|F_{\text{pred}} - F_{\text{GT}}|$', fontsize=LABEL_SIZE)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing between subplots

# Show the combined plot without saving it
plt.show()

print("Individual plots saved, combined plot displayed")