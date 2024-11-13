import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time
import datetime
import os

# Read in data
df = pd.read_csv("musclePBDSim/lM_foce_data.csv")

# Convert data to PyTorch tensors
l_tilde_tensor = torch.tensor(df['lMtilde'].values, dtype=torch.float32)
fM_tensor = torch.tensor(df['force'].values, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(l_tilde_tensor, fM_tensor)

# Split the dataset into training and validation sets
train, valid = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid, batch_size=64)

# Define the MLP model for C(l_tilde)
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, layers=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        for _ in range(layers - 1):
            self.model.append(nn.Linear(hidden_size, hidden_size))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(hidden_size, output_size))

    def forward(self, l_tilde):
        return self.model(l_tilde.unsqueeze(-1)).squeeze(-1)  # Adjusted for single input

# Custom loss function based on the provided formula
def custom_loss(mlp_model, l_tilde, fM_values):
    # Forward pass through the network to get C(l_tilde)
    C_values = mlp_model(l_tilde)

    # Compute the gradient of C with respect to l_tilde
    C_grad_l = torch.autograd.grad(C_values, l_tilde, grad_outputs=torch.ones_like(C_values), create_graph=True)[0]
    
    # Compute the loss according to the formula
    term = C_values * C_grad_l + fM_values
    loss = torch.mean(torch.square(term))  # Fixed this line
    
    return loss

# Instantiate the model and optimizer
model = MLP(hidden_size=128, layers=6)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize TensorBoard writer
runTime = f"{datetime.datetime.fromtimestamp(int(time.time()))}".replace(" ", "_").replace(":", "-")
writer = SummaryWriter(os.path.join("runs_", runTime))

# Training loop
epochs = 600
for epoch in range(epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    for l_tilde, fM in train_loader:
        l_tilde = l_tilde.requires_grad_(True)
        optimizer.zero_grad()
        loss = custom_loss(model, l_tilde, fM)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    valid_loss = 0
    for l_tilde, fM in valid_loader:
        # Enable gradient computation for l_tilde in validation to use custom_loss
        l_tilde = l_tilde.requires_grad_(True)
        # Compute the custom loss, but do not backpropagate
        loss = custom_loss(model, l_tilde, fM)
        valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)
    end_time = time.time()

    # Print loss
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Time: {end_time - start_time:.2f}s')
    
    # Log the losses to TensorBoard
    writer.add_scalars('Losses', {'Training Loss': avg_loss, 'Validation Loss': avg_valid_loss}, epoch)

    # Update the learning rate at the end of each epoch
    scheduler.step()

# Close the TensorBoard writer
writer.close()

# Save the model
torch.save(model.state_dict(), 'musclePBDSim/mlp_model.pth')
