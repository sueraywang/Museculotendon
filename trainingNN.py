import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Generate ActiveForveLength data
x = np.linspace(0.0, 2.0, 1000)
arr_loaded = np.load('ActiveForceLengthData.npy')
force = arr_loaded[0]

# Convert data to PyTorch tensors
length_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
force_tensor = torch.tensor(force, dtype=torch.float32).unsqueeze(1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(length_tensor, force_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = 1
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)

#define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.eval()
with torch.no_grad():
    predicted_force = model(length_tensor).numpy()

# Plot the original vs predicted force-velocity curve
plt.scatter(x, force, alpha=0.5, label="Original")
plt.scatter(x, predicted_force, alpha=0.5, label="Predicted")
plt.title("Original vs Predicted Force-Length Data")
plt.xlabel("Length")
plt.ylabel("Force")
plt.legend()
plt.show()

#save the model
#torch.save(model.state_dict(), 'ActiveForceLengthMLP.pth')

