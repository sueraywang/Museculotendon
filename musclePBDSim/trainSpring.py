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


# Read in data
sample_size = 1000
dx = np.linspace(-2.00, 2.00, sample_size)

# Convert data to PyTorch tensors and reshape for network input
dx_tensor = torch.tensor(dx, dtype=torch.float32).reshape(-1, 1)

# Create a TensorDataset
dataset = TensorDataset(dx_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Create data loaders
train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32)

class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_loss(mlp_model, batch):
    dx = batch[0]
    C_values = mlp_model(dx)
    
    # Scale the terms to help with numerical stability
    term1 = 0.25 * dx**4
    term2 = 0.5 * C_values**2
    
    # Compute scaled residual
    residual = term1 - term2
    loss = torch.mean(residual**2)
    
    return loss

# Set random seed for reproducibility
torch.manual_seed(42)

# Instantiate the model and optimizer
model = MLP(hidden_size=64)
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.2,
    patience=15,
    verbose=True,
    min_lr=1e-6
)

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
        }, 'musclePBDSim/best_model.pth')

# Close the TensorBoard writer
writer.close()

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'valid_loss': avg_valid_loss,
}, 'musclePBDSim/final_model.pth')
