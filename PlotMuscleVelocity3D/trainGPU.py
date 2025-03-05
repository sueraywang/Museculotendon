import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import sys
import os
sys.path.append(os.path.abspath('ComputeMuscleForce'))
from testMuscle import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.version.cuda)  # Should print the CUDA version if PyTorch was built with CUDA support


# Read in data
df = pd.read_csv('PlotMuscleVelocity3D/random3DVelocityData.csv')

# Convert data to PyTorch tensors (on CPU first)
X_train = df[["lMtilde", "lTtilde", "act"]].to_numpy()
y_train = df["vMtilde"].to_numpy()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Keep on CPU for now
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Keep on CPU for now

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train, valid = torch.utils.data.random_split(dataset, [0.8, 0.2])

# Set pin_memory=True only when data is on CPU and will be transferred to GPU during training
train_loader = DataLoader(train, batch_size=64, pin_memory=True, num_workers=0)
valid_loader = DataLoader(valid, batch_size=64, pin_memory=True, num_workers=0)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer=3):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(layer - 1):
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
model = MLP(input_size, hidden_size, output_size, 6).to(device)  # Move model to device

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize TensorBoard writer
runTime = f"{datetime.datetime.fromtimestamp(int(time.time()))}".replace(" ", "_").replace(":", "-")
writer = SummaryWriter(os.path.join("TrainedResults/VelocityMLP3D", runTime))

# Training loop with TensorBoard logging
num_epochs = 600
min_valid_loss = np.inf

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        target = model(inputs)
        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data, labels in valid_loader:
            # Move validation data to GPU
            data, labels = data.to(device), labels.to(device)
            target = model(data)
            loss = criterion(target, labels)
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Time: {end_time - start_time:.2f}s')
    
    # Log the logged average loss to TensorBoard
    writer.add_scalars('Losses', {'Training Loss': avg_loss, 'Validation Loss': avg_valid_loss}, epoch)

    # Update the learning rate at the end of each epoch
    scheduler.step()

# Close the TensorBoard writer
writer.close()

# Save the model
torch.save(model.state_dict(), 'PlotMuscleVelocity3D/mlp_model.pth')

# Check if model and data are on the correct device (GPU)
print("Model is on GPU" if next(model.parameters()).is_cuda else "Model is on CPU")
