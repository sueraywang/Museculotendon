import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import griddata
import sys
sys.path.append('../python-nn')
from testMuscle import *

sample_size = 1000

# # Generate data
# act = 1
# lMtilde = np.random.uniform(.45, 1.85, sample_size)
# lTtilde = np.random.uniform(.99, 1.04, sample_size)
# X, Y = np.meshgrid(lMtilde, lTtilde)
# values = []

# for x, y in zip(X, Y):
#     for x1, y1 in zip(x, y):
#         values.append(calcVelTilde(x1, y1, act, params, curves))

# Z = np.reshape(np.array(values), (-1, sample_size))
# X_train = np.vstack([X.ravel(), Y.ravel()]).T
# y_train = Z.ravel()

# Read in data
DF = pd.read_csv('velocityData_2D_a=0.csv')
# Organize data for plot
X = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:]
Y = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:]
Z = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:]

df = pd.DataFrame()
df = df.assign(X=pd.Series(X.ravel()))
df = df.assign(Y=pd.Series(Y.ravel()))
df = df.assign(Z=pd.Series(Z.ravel()))
df = df[(df["Z"] >= -1.7) & (df["Z"] <= 1.7)]

# Convert data to PyTorch tensors
X_train = df[["X", "Y"]].to_numpy()
y_train = df[["Z"]].to_numpy()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train, valid = torch.utils.data.random_split(dataset,[0.8,0.2])
train_loader = DataLoader(train, batch_size=32)
valid_loader = DataLoader(valid, batch_size=32)

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
input_size = 2
hidden_size = 64
output_size = 1
model = MLP(input_size, hidden_size, output_size, 6)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/twoDVelocityMLP')

# Training loop with TensorBoard logging
num_epochs = 300
min_valid_loss = np.inf

for epoch in range(num_epochs):
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')
    
    # Log the logged average loss to TensorBoard
    writer.add_scalars('Losses', {'Training Loss':avg_loss,'Validation Loss':avg_valid_loss}, epoch)

    # Update the learning rate at the end of each epoch
    scheduler.step()

# Close the TensorBoard writer
writer.close()

# Generate predictions on the training set
with torch.no_grad():
    predicted_velocity = model(X_train_tensor).numpy()
# Reshape results into grid shape
Z_pred = griddata((df['X'], df['Y']), df['Z'], (X, Y), method='cubic')

# Plot the true function vs the model prediction
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.5, 11))
plt.colorbar(label='vMtilde', ticks=np.linspace(-1.5, 1.5, 11)) 
plt.title('True Function')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_pred, levels=np.linspace(-1.5, 1.5, 11))
plt.colorbar(label='vMtilde', ticks=np.linspace(-1.5, 1.5, 11)) 
plt.title('MLP Prediction')

plt.show()

# Save the model
# torch.save(model.state_dict(), 'ActiveForceLengthMLP.pth')