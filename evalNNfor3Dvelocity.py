import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from testMuscle import *
from scipy.interpolate import griddata

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
hidden_size = 64
output_size = 1
model = MLP(input_size, hidden_size, output_size, 6)

# Load the model's state_dict (parameters)
model.load_state_dict(torch.load('mlp_model.pth'))

# Set the model to evaluation mode (important for inference)
model.eval()

# Generate eval data
sample_size = 100
lMtilde = np.linspace(.40, 1.90, sample_size)
lTtilde = np.linspace(.99, 1.07, sample_size)
X, Y = np.meshgrid(lMtilde, lTtilde)
act = [1] * X.ravel().shape[0]
X_train = np.vstack([X.ravel(), Y.ravel(), act]).T
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

with torch.no_grad():  # Disables gradient calculation for faster inference
    predicted_velocity = model(X_train_tensor).numpy()

# Reshape results into grid shape
Z = predicted_velocity.reshape(X.shape)

# Plot the true function vs the model prediction
plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.5, 11))
plt.colorbar(label='vMtilde', ticks=np.linspace(-1.5, 1.5, 11)) 
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.title('MLP Prediction for act = 1')

plt.show()