import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time
import datetime
import os
import sys
sys.path.append('../python-nn')
from testMuscle import *

# Read in data
df = pd.read_csv('threeDimensional/random3DVelocityData.csv')

# Convert data to PyTorch tensors
X_train = df[["lMtilde", "lTtilde", "act"]].to_numpy()
y_train = df["vMtilde"].to_numpy()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train, valid = torch.utils.data.random_split(dataset,[0.8,0.2])
train_loader = DataLoader(train, batch_size=64)
valid_loader = DataLoader(valid, batch_size=64)

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

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize TensorBoard writer
runTime = f"{datetime.datetime.fromtimestamp(int(time.time()))}".replace(" ", "_").replace(":","-")

writer = SummaryWriter(os.path.join("runs_" , runTime))

# Training loop with TensorBoard logging
num_epochs = 600

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        target = model(inputs)
        loss = criterion(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    valid_loss = 0.0
    model.eval()
    for data, labels in valid_loader:  
        target = model(data)
        loss = criterion(target,labels)
        valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Time: {end_time - start_time:.2f}s')
    
    # Log the logged average loss to TensorBoard
    writer.add_scalars('Losses', {'Training Loss':avg_loss,'Validation Loss':avg_valid_loss}, epoch)

    # Update the learning rate at the end of each epoch
    scheduler.step()

# Close the TensorBoard writer
writer.close()

# Save the model
torch.save(model.state_dict(), 'threeDimensional/mlp_model.pth')

# Generate predictions on the training set
#with torch.no_grad():
#    predicted_velocity = model(X_train_tensor).numpy()