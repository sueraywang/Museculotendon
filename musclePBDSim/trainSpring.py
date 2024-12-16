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
import pandas as pd
from Physics import *

"""
df = pd.read_csv("musclePBDSim/dampedLinearSpringData.csv")
dx_tensor = torch.tensor(df['dx'].values, dtype=torch.float32).reshape(-1, 1)
dv_tensor = torch.tensor(df['dv'].values, dtype=torch.float32).reshape(-1, 1)

# Create a TensorDataset
dataset = TensorDataset(dx_tensor, dv_tensor)
# Split the dataset into training and validation sets
train, valid = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train, batch_size=1000, shuffle=True)
valid_loader = DataLoader(valid, batch_size=1000)
"""

# Define constants
sample_size = 1000
dx = np.linspace(-2.00, 2.00, sample_size)
dv = np.linspace(0.00, 0.00, sample_size)  # Example for dv
dx_tensor = torch.tensor(dx, dtype=torch.float32).reshape(-1, 1)
dv_tensor = torch.tensor(dv, dtype=torch.float32).reshape(-1, 1)

# Combine dx and dv into the dataset
dataset = TensorDataset(dx_tensor, dv_tensor)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32)

# Define the modified model
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

# Update the custom loss function
def custom_loss(model, batch):
    dx, dv = batch[0], batch[1]  # Extract inputs
    inputs = torch.cat([dx, dv], dim=1).detach().requires_grad_(True)
    with torch.enable_grad():
        C_values = model(inputs)
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values),
            create_graph=True
        )[0][:, 0]  # Gradient w.r.t dx (first input)
        residual = C_values * C_grad - dx  # Example residual, modify as needed
        loss = torch.mean(residual**2)
    return loss

# Instantiate the enhanced MLP
model = MLP(input_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize TensorBoard writer
runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(os.path.join("runs_", runTime))

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
    scheduler.step()
    
    # Save best model
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
        }, 'musclePBDSim/springForceBestModel_withDamping.pth')

# Close the TensorBoard writer
writer.close()

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'valid_loss': avg_valid_loss,
}, 'musclePBDSim/springForceFinalModel_withDamping.pth')
