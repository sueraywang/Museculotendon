import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Generate ActiveForceLength data
x = np.linspace(0.0, 2.0, 1000)
arr_loaded = np.load('ActiveForceLengthData.npy')
force = arr_loaded[0]

# Convert data to PyTorch tensors
length_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
force_tensor = torch.tensor(force, dtype=torch.float32).unsqueeze(1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(length_tensor, force_tensor)
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
input_size = 1
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size, 6)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/ActiveForceLengthMLP')

# Training loop with TensorBoard logging
num_epochs = 100
min_valid_loss = np.inf

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()     # Optional when not using Model Specific layer
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        target = model(inputs)
        loss = criterion(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in valid_loader:  
        target = model(data)
        loss = criterion(target,labels)
        valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')
    
    # Log the logged average loss to TensorBoard
    writer.add_scalar('Logged Training Loss', np.log(avg_loss), epoch)
    writer.add_scalar('Logged Validation Loss', np.log(avg_valid_loss), epoch)

# Close the TensorBoard writer
writer.close()

with torch.no_grad():
    predicted_force = model(length_tensor).numpy()

# Plot the original vs predicted force-length curve
plt.scatter(x, force, alpha=0.5, label="Original")
plt.scatter(x, predicted_force, alpha=0.5, label="Predicted")
plt.title("Original vs Predicted Force-Length Data")
plt.xlabel("Length")
plt.ylabel("Force")
plt.legend()
plt.show()

# Save the model
# torch.save(model.state_dict(), 'ActiveForceLengthMLP.pth')
