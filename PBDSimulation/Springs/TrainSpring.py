import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import os
from Physics import *

# Define the model (same as in your script)
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
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

# Enhanced data generation with importance sampling and zero point constraint
def generate_batch(batch_size, x_min=-2.0, x_max=2.0, focus_ratio=0.5, include_zero=True):
    # Adjust batch size to make room for zero point if needed
    effective_batch_size = batch_size - (1 if include_zero else 0)
    
    # Split batch into uniform sampling and focused sampling
    uniform_size = int(effective_batch_size * (1 - focus_ratio))
    focused_size = effective_batch_size - uniform_size
    
    # Generate uniform samples across the entire range
    uniform_samples = torch.FloatTensor(uniform_size, 1).uniform_(x_min, x_max)
    
    # Generate focused samples - these regions are often more challenging for the cubic spring
    # Near zero (small values)
    near_zero_samples = torch.FloatTensor(focused_size // 3, 1).uniform_(-0.5, 0.5)
    # Negative extreme values
    extreme_neg_samples = torch.FloatTensor(focused_size // 3, 1).uniform_(x_min, x_min/2)
    # Positive extreme values
    extreme_pos_samples = torch.FloatTensor(focused_size - (focused_size // 3) * 2, 1).uniform_(x_max/2, x_max)
    
    # Combine all samples
    focused_samples = torch.cat([near_zero_samples, extreme_neg_samples, extreme_pos_samples], dim=0)
    
    # Combine uniform and focused samples
    dx = torch.cat([uniform_samples, focused_samples], dim=0)
    
    # Add exact zero point if needed
    if include_zero:
        zero_point = torch.zeros(1, 1)
        dx = torch.cat([dx, zero_point], dim=0)
    
    # Shuffle to mix uniform and focused samples
    idx = torch.randperm(dx.size(0))
    dx = dx[idx]
    
    return [dx]

# Enhanced custom loss function with zero-point constraint
def custom_loss(model, batch, zero_weight=5.0):
    dx = batch[0]  # Extract inputs
    inputs = dx.detach().requires_grad_(True)
    with torch.enable_grad():
        C_values = model(inputs)
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values),
            create_graph=True
        )[0]  # Gradient w.r.t dx
        
        # Modified residual for x^3 relationship
        x_cubed = dx[:, 0]**3  # x^3
        residual = C_values[:, 0] * C_grad[:, 0] - x_cubed  # C(x)*dC(x) - x^3
        
        # Find points that are exactly or very close to zero
        zero_mask = torch.abs(dx[:, 0]) < 1e-6
        
        # Regular loss for all points
        general_loss = torch.mean(residual**2)
        
        # Additional loss specifically for the zero point constraint
        if zero_mask.any():
            zero_loss = torch.mean((C_values[zero_mask, 0] * C_grad[zero_mask, 0])**2)
            # Apply higher weight to the zero point loss
            total_loss = general_loss + zero_weight * zero_loss
        else:
            total_loss = general_loss
            
        return total_loss

# Initialize model, optimizer, and scheduler as in your script
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# Initialize TensorBoard writer
runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("TrainedResults/CubicSpring", runTime)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Create a fixed validation set for consistent evaluation
valid_batch_size = 2000  # Larger validation set for better accuracy
valid_batch = generate_batch(valid_batch_size, focus_ratio=0, include_zero=True)  # Uniform sampling for validation

# Training parameters
epochs = 10000
batch_size = 1000
batches_per_epoch = 30
best_valid_loss = float('inf')
target_error = 1e-7  # Target error threshold
patience = 50

# Initialize early stopping variables
below_threshold_count = 0
epochs_without_improvement = 0

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    
    # Generate and train on multiple batches per epoch
    for _ in range(batches_per_epoch):
        # Generate fresh data with importance sampling and include the zero point
        batch = generate_batch(batch_size, focus_ratio=0.0, include_zero=True)
        
        optimizer.zero_grad(set_to_none=True)
        loss = custom_loss(model, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / batches_per_epoch

    # Validation phase - use the fixed validation batch
    model.eval()
    with torch.no_grad():
        valid_loss = custom_loss(model, valid_batch).item()
    
    end_time = time.time()

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {avg_train_loss:.8f}, '
          f'Validation Loss: {valid_loss:.8f}, '
          f'Time: {end_time - start_time:.2f}s')
    
    # Log the losses to TensorBoard
    writer.add_scalars('Losses', {
        'Training': avg_train_loss,
        'Validation': valid_loss
    }, epoch)

    # Learning rate scheduling
    scheduler.step(valid_loss)
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_dir = "TrainedModels"
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'valid_loss': valid_loss,
        }, os.path.join(save_dir, 'cubic_spring_best_focused_model.pth'))
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    # Check if validation loss is below target error
    if valid_loss < target_error:
        below_threshold_count += 1
        print(f"Validation loss below target error for {below_threshold_count} consecutive epochs")
        
        # If validation loss has been below threshold for 5 consecutive epochs, stop training
        if below_threshold_count >= 5:
            print(f"Early stopping: Target error {target_error} reached for 5 consecutive epochs")
            break
    else:
        below_threshold_count = 0
    
    # Stop if no improvement for 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping: No improvement for {patience} epochs")
        break

# Close the TensorBoard writer
writer.close()

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'valid_loss': valid_loss,
    'stopped_early': epoch < epochs - 1,
    'reached_target': valid_loss < target_error,
}, os.path.join("TrainedModels", 'cubic_spring_final_focused_model.pth'))

print(f"Training completed after {epoch+1} epochs")
if valid_loss < target_error:
    print(f"Target error of {target_error} achieved!")
else:
    print(f"Best validation loss: {best_valid_loss}")