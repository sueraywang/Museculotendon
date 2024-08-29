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
force = arr_loaded[1]

# Convert data to PyTorch tensors
length_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
force_tensor = torch.tensor(force, dtype=torch.float32).unsqueeze(1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(length_tensor, force_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

model = MLP(input_size, hidden_size, output_size, 20)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/ActiveForceLengthMLP')

# Training loop with TensorBoard logging
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Log the average loss to TensorBoard
    writer.add_scalar('Training Loss', avg_loss, epoch)

# After training, evaluate the model
model.eval()
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

# Close the TensorBoard writer
writer.close()
