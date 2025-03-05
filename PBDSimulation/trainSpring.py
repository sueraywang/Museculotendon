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
from Physics import *

sample_size = 1000
dx = np.linspace(-2.00, 2.00, sample_size)
dx_tensor = torch.tensor(dx, dtype=torch.float32).reshape(-1, 1)

# Combine dx into the dataset
dataset = TensorDataset(dx_tensor)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train, batch_size=1000, shuffle=True)
valid_loader = DataLoader(valid, batch_size=1000)

# Define the modified model
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
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

# Update the custom loss function for x^3 relationship
def custom_loss(model, batch):
    dx = batch[0]  # Extract inputs
    inputs = torch.cat([dx], dim=1).detach().requires_grad_(True)
    with torch.enable_grad():
        C_values = model(inputs)
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values),
            create_graph=True
        )[0][:, 0]  # Gradient w.r.t dx
        
        # Modified residual for x^3 relationship
        x_cubed = dx[:, 0]**3  # x^3
        residual = C_values[:, 0] * C_grad - x_cubed  # C(x)*dC(x) - x^3
        
        loss = torch.mean(residual**2)
    return loss

# Instantiate the enhanced MLP
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# Initialize TensorBoard writer
runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(os.path.join("TrainedResults/CubicSpring", runTime))

# Training loop
epochs = 1000
best_valid_loss = float('inf')

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

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {avg_train_loss:.8f}, '
          f'Validation Loss: {avg_valid_loss:.8f}, '
          f'Time: {end_time - start_time:.2f}s')
    
    # Log the losses to TensorBoard
    writer.add_scalars('Losses', {
        'Training': avg_train_loss,
        'Validation': avg_valid_loss
    }, epoch)

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
        }, 'TrainedModels/cubic_spring_best_model.pth')

# Close the TensorBoard writer
writer.close()

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'valid_loss': avg_valid_loss,
}, 'TrainedModels/cubic_spring_final_model.pth')

import matplotlib.pyplot as plt
# Add visualization code
print("Creating visualizations...")

# Create test data spanning the full range
test_dx = torch.linspace(-2.0, 2.0, 500).reshape(-1, 1)

# For predictions without gradients
model.eval()
with torch.no_grad():
    test_C = model(test_dx).cpu().numpy()
    true_force = test_dx.cpu().numpy()**3  # The true force is now x^3

# For gradient calculations
grads = []
for i in range(len(test_dx)):
    # Create a fresh tensor for each gradient calculation
    single_input = test_dx[i:i+1].clone().detach().requires_grad_(True)
    single_output = model(single_input)
    single_grad = torch.autograd.grad(
        outputs=single_output, 
        inputs=single_input,
        grad_outputs=torch.ones_like(single_output)
    )[0].item()
    grads.append(single_grad)

# Generate plot for spring simulation
plt.figure(figsize=(12, 10))

# Plot 1: Predicted C vs True Force
plt.subplot(3, 1, 1)
computed_force = np.array(grads) * test_C.flatten()
plt.plot(test_dx.numpy(), computed_force, 'b-', label='C(x)*dC/dx (computed)')
plt.plot(test_dx.numpy(), true_force, 'r--', label='x^3 (target)')
plt.grid(True)
plt.legend()
plt.title('Predicted C vs x^3')
plt.xlabel('x')
plt.ylabel('Value')

# Plot 2: Gradient of C vs Expected Gradient
plt.subplot(3, 1, 2)
plt.plot(test_dx.numpy(), grads, 'b-', label='dC/dx')
expected_grad = test_C / test_dx.numpy()**3
plt.plot(test_dx.numpy(), expected_grad, 'r--', label='C/x^3')
plt.grid(True)
plt.legend()
plt.title('Gradient of C vs C/x^3')
plt.xlabel('x')
plt.ylabel('Value')

# Plot 3: Residual - how close are we to satisfying the constraint?
plt.subplot(3, 1, 3)
residual = np.array(grads) * test_C.flatten() - true_force.flatten()
plt.plot(test_dx.numpy(), residual, 'g-', label='C*dC/dx - x^3')
plt.grid(True)
plt.legend()
plt.title('Residual (Should be close to zero)')
plt.xlabel('x')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig('TrainedResults/CubicSpring/results.png')
plt.show()