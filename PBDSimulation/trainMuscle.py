import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import griddata

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

print(f"Data shape: {inputs.shape[0]} samples with {inputs.shape[1]} features")
print(f"Output shape: {outputs.shape}")

# Convert to tensor format
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

# Combine inputs and outputs into the dataset
dataset = TensorDataset(inputs_tensor, outputs_tensor)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Increase batch size for faster training since we're processing simpler data
batch_size = 128
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=batch_size)

# Define the model for 2D input
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

# Updated custom loss function with simplified gradient calculation
def custom_loss(model, batch):
    x = batch[0]  # Input tensor [lMtilde, activation]
    target_f = batch[1]  # f(x) target values
    
    # Make input require gradient for computing dC/dx (wrt lMtilde)
    inputs = x.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Forward pass to get C(x)
        C_values = model(inputs)
        
        # Compute gradient dC/dx with respect to lMtilde (first input column)
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values),
            create_graph=True
        )[0]
        
        # Extract only the gradient with respect to lMtilde (first column)
        lMtilde_grad = C_grad[:, 0].view(-1, 1)
        
        # Compute C(x)*dC/dx = -f(x) relationship
        left_side = C_values * lMtilde_grad  # C(x)*dC/dx
        right_side = -target_f  # -f(x)
        
        residual = left_side - right_side  # C(x)*dC/dx - (-f(x))
        
        loss = torch.mean(residual**2)
    return loss

# Instantiate the MLP
model = MLP(input_size=2)  # Takes 2 inputs: lMtilde and activation
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# Initialize TensorBoard writer with a descriptive name
runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(os.path.join("TrainedResults/MuscleWithAct", runTime))

# Training loop
epochs = 600
best_valid_loss = float('inf')

# Track time for each epoch
epoch_times = []

for epoch in range(epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = custom_loss(model, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            loss = custom_loss(model, batch)
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)
    
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {avg_train_loss:.8f}, '
          f'Validation Loss: {avg_valid_loss:.8f}, '
          f'Time: {epoch_time:.2f}s')
    
    # Log the losses to TensorBoard
    writer.add_scalars('Losses', {
        'Training': avg_train_loss,
        'Validation': avg_valid_loss
    }, epoch)
    writer.add_scalar('Epoch Time', epoch_time, epoch)

    # Learning rate scheduling
    scheduler.step(avg_valid_loss)
    
    # Save best model
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
        }, 'TrainedModels/muscle_length_and_act_best_model.pth')
        
    # Early stopping if validation loss is extremely small
    if avg_valid_loss < 1e-6:
        print(f"Validation loss {avg_valid_loss:.10f} is very small. Early stopping.")
        break

# Print training statistics
avg_epoch_time = sum(epoch_times) / len(epoch_times)
print(f"Average epoch time: {avg_epoch_time:.2f}s")
print(f"Best validation loss: {best_valid_loss:.10f}")

# Close the TensorBoard writer
writer.close()

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'valid_loss': avg_valid_loss,
}, 'TrainedModels/muscle_length_and_act_final_model.pth')

# Add visualization code
print("Creating visualizations...")

# Create a grid of test points for visualization
lm_min, lm_max = np.min(inputs[:, 0]), np.max(inputs[:, 0])
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

# Convert original data to tensors for model prediction
original_inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# Get model predictions for the original data points
with torch.no_grad():
    predicted_force = model(original_inputs_tensor).cpu().numpy().flatten()

# Calculate gradients for the original data points
original_grads = np.zeros((inputs.shape[0], 1))
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

# Interpolate errors onto the grid
grid_z = griddata((inputs[:, 0], inputs[:, 1]), abs_error, 
                  (lm_grid, act_grid), method='linear')

# Set up LaTeX rendering
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Create combined figure with 4 subplots
plt.figure(figsize=(20, 16))

# Plot 1: Learned C(x) Function as a 3D surface
ax1 = plt.subplot(221, projection='3d')
surf1 = ax1.plot_surface(lm_grid, act_grid, C_reshaped, cmap='viridis', alpha=0.8)
ax1.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=14)
ax1.set_ylabel(r'Activation ($a$)', fontsize=14)
ax1.set_zlabel(r'$C(x)$', fontsize=14)
ax1.set_title(r'Learned Function $C(x)$', fontsize=16)
cbar1 = plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot 2: Gradient dC/dlM as a 3D surface
ax2 = plt.subplot(222, projection='3d')
surf2 = ax2.plot_surface(lm_grid, act_grid, grads_reshaped, cmap='plasma', alpha=0.8)
ax2.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=14)
ax2.set_ylabel(r'Activation ($a$)', fontsize=14)
ax2.set_zlabel(r'$\frac{\partial C}{\partial \tilde{l}_M}$', fontsize=16)
ax2.set_title(r'Gradient $\frac{\partial C}{\partial \tilde{l}_M}$', fontsize=16)
cbar2 = plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Plot 3: Computed Force (C*dC/dlM) as a 3D surface
ax3 = plt.subplot(223, projection='3d')
surf3 = ax3.plot_surface(lm_grid, act_grid, -computed_force_reshaped, cmap='coolwarm', alpha=0.8)
ax3.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=14)
ax3.set_ylabel(r'Activation ($a$)', fontsize=14)
ax3.set_zlabel(r'Force $\left(-C \cdot \frac{\partial C}{\partial \tilde{l}_M}\right)$', fontsize=16)
ax3.set_title(r'Computed Muscle Force $\left(-C \cdot \frac{\partial C}{\partial \tilde{l}_M}\right)$', fontsize=16)
cbar3 = plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# Plot 4: Error Contour Plot
ax4 = plt.subplot(224)
contour = ax4.contourf(lm_grid, act_grid, grid_z, 20, cmap='viridis')
ax4.set_xlabel(r'$\tilde{l}_M$ (Normalized Muscle Length)', fontsize=14)
ax4.set_ylabel(r'Activation ($a$)', fontsize=14)
ax4.set_title(r'Absolute Error $|F_{\text{pred}} - F_{\text{GT}}|$', fontsize=16)
cbar4 = plt.colorbar(contour, ax=ax4, shrink=0.5, aspect=5)

# Add scatter points to error contour
scatter = ax4.scatter(inputs[:, 0], inputs[:, 1], c=abs_error, 
                     cmap='viridis', s=5, alpha=0.2, edgecolors='none')

plt.tight_layout()
plt.savefig('TrainedResults/MuscleWithAct/combined_visualization.png', dpi=300, bbox_inches='tight')

print("Visualization complete!")