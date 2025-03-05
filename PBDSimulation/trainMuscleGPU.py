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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load and prepare data with three inputs
def prepare_data(file_path, batch_size=128):
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    lMtilde = data['L']        # M×N array
    activation = data['A']     # M×N array
    pennation = data['P']      # M×N array - new input
    muscle_force = data['F']   # M×N array

    # Flatten all arrays directly
    lMtilde_flat = lMtilde.flatten()
    activation_flat = activation.flatten()
    pennation_flat = pennation.flatten()
    muscle_force_flat = muscle_force.flatten()

    # Create input array with shape (num_samples, 3) - now with 3 inputs
    inputs = np.column_stack((lMtilde_flat, activation_flat, pennation_flat))
    outputs = muscle_force_flat.reshape(-1, 1)

    print(f"Data shape: {inputs.shape[0]} samples with {inputs.shape[1]} features")
    print(f"Output shape: {outputs.shape}")
    print(f"Input range - lMtilde: [{np.min(lMtilde_flat):.4f}, {np.max(lMtilde_flat):.4f}], " +
          f"activation: [{np.min(activation_flat):.4f}, {np.max(activation_flat):.4f}], " +
          f"pennation: [{np.min(pennation_flat):.4f}, {np.max(pennation_flat):.4f}]")
    print(f"Output range - force: [{np.min(muscle_force_flat):.4f}, {np.max(muscle_force_flat):.4f}]")

    # Convert to tensor format and move to GPU
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32, device=device)

    # Combine inputs and outputs into the dataset
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Create dataloaders with pinned memory for faster transfers to GPU
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=False)
    valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=False)
    
    return train_loader, valid_loader

# Define the model for 3D input (updated input_size default)
class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=1):
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

# Updated custom loss function optimized for GPU - still using gradient wrt lMtilde
def custom_loss(model, batch):
    x, target_f = batch
    
    # Make input require gradient for computing dC/dx (wrt lMtilde)
    inputs = x.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Forward pass to get C(x)
        C_values = model(inputs)
        
        # Compute gradient dC/dx with respect to lMtilde (first input column)
        # We use create_graph=True because we need to backpropagate through this gradient
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values, device=device),
            create_graph=True
        )[0]
        
        # Extract only the gradient with respect to lMtilde (first column)
        # Still using only the first column (lMtilde) for the constraint
        lMtilde_grad = C_grad[:, 0].view(-1, 1)
        
        # Compute C(x)*dC/dx = -f(x) relationship
        left_side = C_values * lMtilde_grad  # C(x)*dC/dx
        right_side = -target_f  # -f(x)
        
        residual = left_side - right_side  # C(x)*dC/dx - (-f(x))
        
        loss = torch.mean(residual**2)
    return loss

def train_model(model, train_loader, valid_loader, epochs=1000, lr=0.001, result_dir='TrainedResults', model_dir='TrainedModels'):
    # Move model to GPU
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Initialize TensorBoard writer with a descriptive name
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(result_dir, "MuscleWithActAndPennation", f"gpu_{runTime}"))

    # Training metrics
    best_valid_loss = float('inf')
    epoch_times = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model architecture:\n{model}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()
            
            # Forward pass with the custom loss
            loss = custom_loss(model, batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
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
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.8f}, '
              f'Validation Loss: {avg_valid_loss:.8f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Log the losses to TensorBoard
        writer.add_scalars('Losses', {
            'Training': avg_train_loss,
            'Validation': avg_valid_loss
        }, epoch)
        writer.add_scalar('Epoch Time', epoch_time, epoch)

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
            }, os.path.join(model_dir, 'gpu_muscle_3inputs_best_model.pth'))
            
        # Early stopping if validation loss is extremely small
        if avg_valid_loss < 1e-6:
            print(f"Validation loss {avg_valid_loss:.10f} is very small. Early stopping.")
            break

        # Report GPU memory usage if using CUDA
        if device.type == 'cuda':
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
            writer.add_scalar('GPU/Memory_Allocated_MB', gpu_memory_allocated, epoch)
            writer.add_scalar('GPU/Memory_Reserved_MB', gpu_memory_reserved, epoch)
            
            if epoch % 10 == 0:  # Print every 10 epochs to keep logs cleaner
                print(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")

    # Print training statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Training completed in {len(epoch_times)} epochs")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    print(f"Best validation loss: {best_valid_loss:.10f}")

    # Close the TensorBoard writer
    writer.close()

    # Save the final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'valid_loss': avg_valid_loss,
    }, os.path.join(model_dir, 'gpu_muscle_3inputs_final_model.pth'))
    
    return model, best_valid_loss, epoch_times

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train muscle model on GPU with three inputs')
    parser.add_argument('--data', type=str, default='muscle_force_surface.npz', help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--model_dir', type=str, default='TrainedModels', help='Output directory for models')
    parser.add_argument('--result_dir', type=str, default='TrainedResults', help='Output directory for training metrics')
    
    args = parser.parse_args()
    
    # Prepare data
    train_loader, valid_loader = prepare_data(args.data, args.batch_size)
    
    # Create model with specified hidden size, now with 3 inputs
    model = MLP(input_size=3, hidden_size=args.hidden_size, output_size=1)
    
    # Train model
    model, best_loss, epoch_times = train_model(
        model, 
        train_loader, 
        valid_loader, 
        epochs=args.epochs,
        lr=args.lr,
        model_dir=args.model_dir,
        result_dir=args.result_dir
    )
    
    print(f"Training complete! Best validation loss: {best_loss:.10f}")
    print(f"Models saved in {args.model_dir}")