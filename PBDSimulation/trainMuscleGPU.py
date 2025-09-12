import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import random
import time
import datetime
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Physics import params, model_params

# Model definition
class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=1, num_layers=3, activation='relu'):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Create layers dynamically
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for i in range(num_layers - 2):  # -2 because we have input and output layers
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
        # Set activation function
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.elu  # Default
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # All layers except the last one get activation
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

# Function that calculates muscle force
def muscle_force(lMtilde: float, act: float, vMtilde: float) -> float:
    afl = params.curve_afl.calc_value(lMtilde)
    pfl = params.curve_pfl.calc_value(lMtilde)
    fv = params.curve_fv.calc_value(vMtilde)
    
    return act * afl * fv + pfl# + params.beta * vMtilde

# Function that calculates df/dl during data generation
def muscle_force_derivative(lMtilde: float, act: float, vMtilde: float) -> np.ndarray:
    """
    Compute the analytical gradient of the muscle function
    Returns: [dF/dlMtilde, dF/dvMtilde]
    """
    # Get values and derivatives from curves
    res_afl = params.curve_afl.calc_val_deriv(lMtilde)
    afl = res_afl[0]
    afl_deriv = res_afl[1]  # derivative w.r.t lMtilde
    
    pfl_deriv = params.curve_pfl.calc_derivative(lMtilde, 1) # derivative w.r.t lMtilde
    
    res_fv = params.curve_fv.calc_val_deriv(vMtilde)
    fv = res_fv[0]
    fv_deriv = res_fv[1]  # derivative w.r.t vMtilde
    
    # Compute partial derivatives
    # ∂F/∂lMtilde = fMopt * (act * (∂afl/∂lMtilde * fv) + ∂pfl/∂lMtilde)
    dF_dlMtilde = act * afl_deriv * fv + pfl_deriv
    
    # ∂F/∂vMtilde = fMopt * (act * afl * ∂fv/∂vMtilde + beta)
    dF_dvMtilde = act * afl * fv_deriv + params.beta
    
    return np.array([dF_dlMtilde, dF_dvMtilde])

# Custom loss function with three physics terms
def custom_physics_loss(targets, dF_dl_targets, dF_dv_targets, inputs, model, physics_weight1=1.0, physics_weight2=1.0, ENABLE_GRAD=False):
    """
    Loss function with three physics terms:
    1. Primary: f + C * dC_dl = 0
    2. Secondary: df/dl + (dC_dl * dC_dl + C * d2C_dldl) = 0  
    3. Third: df/dv + (dC_dv * dC_dl + C * d2C_dldv) = 0
    
    Args:
        targets: ground truth force values
        dF_dl_targets: pre-computed df/dl values from data generation
        dF_dv_targets: pre-computed df/dv values from data generation
        inputs: input tensor [lMtilde, act, vMtilde]
        model: neural network model (constraint C)
        physics_weight1: weight for the secondary physics regularization term
        physics_weight2: weight for the third physics regularization term
    """
    # Get input components
    lMtilde = inputs[:, 0].unsqueeze(1)
    act = inputs[:, 1].unsqueeze(1)
    vMtilde = inputs[:, 2].unsqueeze(1)
    
    # Create inputs that require gradients for both lMtilde and vMtilde
    lMtilde_with_grad = lMtilde.detach().requires_grad_(True)
    vMtilde_with_grad = vMtilde.detach().requires_grad_(True)
    
    # Reconstruct input tensor with gradient-enabled variables
    new_inputs = torch.cat([lMtilde_with_grad, act, vMtilde_with_grad], dim=1)
    
    # Forward pass to get C
    C = model(new_inputs)
    
    # Calculate first derivatives
    dC_dl = torch.autograd.grad(
        outputs=C,
        inputs=lMtilde_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    dC_dv = torch.autograd.grad(
        outputs=C,
        inputs=vMtilde_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Calculate second derivatives
    d2C_dldl = torch.autograd.grad(
        outputs=dC_dl,
        inputs=lMtilde_with_grad,
        grad_outputs=torch.ones_like(dC_dl),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Calculate mixed derivative d²C/dl∂v
    d2C_dldv = torch.autograd.grad(
        outputs=dC_dl,
        inputs=vMtilde_with_grad,
        grad_outputs=torch.ones_like(dC_dl),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # PRIMARY LOSS: f + C * dC_dl = 0
    physics_pred_primary = C * dC_dl
    loss_primary = F.mse_loss(physics_pred_primary, -targets)
    if ENABLE_GRAD is True:
        # SECONDARY LOSS: df/dl + (dC_dl * dC_dl + C * d2C_dldl) = 0
        physics_pred_secondary = dC_dl * dC_dl + C * d2C_dldl
        loss_secondary = F.mse_loss(physics_pred_secondary, -dF_dl_targets)
        
        # THIRD LOSS: df/dv + (dC_dv * dC_dl + C * d2C_dldv) = 0
        physics_pred_third = dC_dv * dC_dl + C * d2C_dldv
        loss_third = F.mse_loss(physics_pred_third, -dF_dv_targets)
        
        # Combine all terms
        total_loss = loss_primary + physics_weight1 * loss_secondary + physics_weight2 * loss_third
    else:
        # Combine all terms
        total_loss = loss_primary
    
    return total_loss

# Training function with early stopping and best model saving
def train_model(model, train_loader, valid_loader, num_epochs=100, device='cuda', 
                        use_tensorboard=False, physics_weight1=1.0, physics_weight2=1.0, enable_grad=False,
                        optimizer_type='adamw', scheduler_type='plateau', learning_rate=0.001):
    """Training function with more optimizer and scheduler options"""
    
    model = model.to(device)
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Choose optimizer
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Choose scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if use_tensorboard:
        log_dir = os.path.join('TrainedResults/Muscles/LenActVel', runTime)
        writer = SummaryWriter(log_dir)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    # Calculate total number of iterations
    total_iterations = num_epochs * len(train_loader)
    
    # Create a single progress bar for the entire training process
    pbar = tqdm(total=total_iterations, desc="Training Progress")
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 200
    
    # Record start time for timing
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets, dF_dl_targets, dF_dv_targets in train_loader:
            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            dF_dl_targets = dF_dl_targets.to(device)
            dF_dv_targets = dF_dv_targets.to(device)
            
            optimizer.zero_grad()
            
            # Use your original custom physics-based loss function
            loss = custom_physics_loss(targets, dF_dl_targets, dF_dv_targets, inputs, model, physics_weight1, physics_weight2, enable_grad)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update scheduler if it's step-based
            if scheduler_type == 'cosine':
                scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Update progress bar
            pbar.update(1)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, dF_dl_targets, dF_dv_targets in valid_loader:
                # Move data to GPU
                inputs = inputs.to(device)
                targets = targets.to(device)
                dF_dl_targets = dF_dl_targets.to(device)
                dF_dv_targets = dF_dv_targets.to(device)
                
                # During validation, we still use the custom loss but without creating computation graphs
                with torch.enable_grad():
                    inputs_detached = inputs.detach()
                    targets_detached = targets.detach()
                    dF_dl_targets_detached = dF_dl_targets.detach()
                    dF_dv_targets_detached = dF_dv_targets.detach()
                    loss = custom_physics_loss(targets_detached, dF_dl_targets_detached, dF_dv_targets_detached, inputs_detached, model, physics_weight1, physics_weight2, enable_grad)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(valid_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        elif scheduler_type in ['step', 'exponential']:
            scheduler.step()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log losses and timing to TensorBoard if enabled
        if use_tensorboard and writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalars('Losses/train_vs_validation', {
                'train': train_loss,
                'validation': val_loss
            }, epoch)
            writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Training/elapsed_time_minutes', elapsed_time / 60, epoch)
        
        # Update progress bar description
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            os.makedirs('TrainedModels/Muscles', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': val_loss,
                'model_config': {
                    'input_size': model.input_size,
                    'hidden_size': model.hidden_size,
                    'output_size': model.output_size,
                    'num_layers': model.num_layers,
                }
            }, os.path.join('TrainedModels/Muscles', model_params['model_name']))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                pbar.set_description(f"Early stopping at epoch {epoch+1}")
                break
    
    # Close the progress bar
    pbar.close()
    
    # Log final training time
    if use_tensorboard and writer is not None:
        total_time = time.time() - start_time
        writer.add_scalar('Training/total_time_minutes', total_time / 60, 0)
        writer.close()
    
    return train_losses, val_losses

# Generate synthetic data using the muscle_force function
def generate_synthetic_data(num_samples=10000):
    # Define ranges for input variables
    lMtilde_min, lMtilde_max = 1e-6, 2.0  # Example range for normalized muscle length
    act_min, act_max = 0.0, 1.0  # Activation typically ranges from 0 to 1
    vMtilde_min, vMtilde_max = -2.0, 2.0  # Pennation angle in radians
    
    # Generate random input values
    lMtilde_values = np.random.uniform(lMtilde_min, lMtilde_max, num_samples)
    act_values = np.random.uniform(act_min, act_max, num_samples)
    vMtilde_values = np.random.uniform(vMtilde_min, vMtilde_max, num_samples)
    
    # Calculate corresponding muscle forces and derivatives
    force_values = np.zeros(num_samples)
    dF_dl_values = np.zeros(num_samples)
    dF_dv_values = np.zeros(num_samples)
    
    for i in range(num_samples):
        force_values[i] = muscle_force(lMtilde_values[i], act_values[i], vMtilde_values[i])
        derivatives = muscle_force_derivative(lMtilde_values[i], act_values[i], vMtilde_values[i])
        dF_dl_values[i] = derivatives[0]  # dF/dlMtilde
        dF_dv_values[i] = derivatives[1]  # dF/dvMtilde
    
    # Create input and output arrays
    inputs = np.column_stack((lMtilde_values, act_values, vMtilde_values))
    outputs = force_values.reshape(-1, 1)
    dF_dl_outputs = dF_dl_values.reshape(-1, 1)
    dF_dv_outputs = dF_dv_values.reshape(-1, 1)
    
    return inputs, outputs, dF_dl_outputs, dF_dv_outputs

def generate_focused_data_method(num_samples=10000):
    # Define ranges for input variables
    lMtilde_min, lMtilde_max = 1e-6, 2.0  # Example range for normalized muscle length
    act_min, act_max = 0.0, 1.0  # Activation typically ranges from 0 to 1
    vMtilde_min, vMtilde_max = -2.0, 2.0  # Pennation angle in radians
    
    # Generate random input values
    lMtilde_values = np.random.uniform(lMtilde_min, lMtilde_max, num_samples)
    act_values = np.random.uniform(act_min, act_max, num_samples)
    
    # Focus around zero velocity (normal distribution)
    # Calculate exact numbers
    n1 = int(num_samples * 0.8)
    n2 = int(num_samples * 0.15)
    n3 = num_samples - n1 - n2  # This ensures the total equals exactly num_samples

    # Generate and combine
    values1 = np.random.uniform(vMtilde_min, vMtilde_max, n1)
    values2 = np.random.uniform(-0.3, 0.3, n2)
    values3 = np.zeros(n3)
    vMtilde_values = np.concatenate([values1, values2, values3])
    
    # Calculate corresponding muscle forces and derivatives
    force_values = np.zeros(num_samples)
    dF_dl_values = np.zeros(num_samples)
    dF_dv_values = np.zeros(num_samples)  # New array for dF/dvMtilde
    
    for i in range(num_samples):
        force_values[i] = muscle_force(lMtilde_values[i], act_values[i], vMtilde_values[i])
        derivatives = muscle_force_derivative(lMtilde_values[i], act_values[i], vMtilde_values[i])
        dF_dl_values[i] = derivatives[0]  # dF/dlMtilde
        dF_dv_values[i] = derivatives[1]  # dF/dvMtilde
    
    # Create input and output arrays
    inputs = np.column_stack((lMtilde_values, act_values, vMtilde_values))
    outputs = force_values.reshape(-1, 1)
    dF_dl_outputs = dF_dl_values.reshape(-1, 1)
    dF_dv_outputs = dF_dv_values.reshape(-1, 1)  # New output
    
    return inputs, outputs, dF_dl_outputs, dF_dv_outputs


# Data loading function
def load_and_prepare_data(batch_size, num_samples=100000):
    # Generate synthetic data (now returns 4 components)
    inputs, outputs, dF_dl_outputs, dF_dv_outputs = generate_focused_data_method(num_samples)
    
    print(f"Generated data shape: {inputs.shape[0]} samples with {inputs.shape[1]} features")
    print(f"Output shape: {outputs.shape}")
    print(f"dF/dl shape: {dF_dl_outputs.shape}")
    print(f"dF/dv shape: {dF_dv_outputs.shape}")
    
    # Convert to tensor format
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    dF_dl_tensor = torch.tensor(dF_dl_outputs, dtype=torch.float32)
    dF_dv_tensor = torch.tensor(dF_dv_outputs, dtype=torch.float32)
    
    # Create dataset with four components: inputs, targets, dF_dl, and dF_dv
    dataset = TensorDataset(inputs_tensor, outputs_tensor, dF_dl_tensor, dF_dv_tensor)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
    
    return train_loader, valid_loader

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train muscle force prediction model')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of synthetic data samples to generate')
    parser.add_argument('--physics_weight1', type=float, default=1.0, help='Weight for first physics regularization term')
    parser.add_argument('--physics_weight2', type=float, default=1.0, help='Weight for second physics regularization term')
    parser.add_argument('--enable_grad_loss', type=float, default=False, help='Ask loss function to include gradient minimization')
    
    # Configurable number of layers
    parser.add_argument('--num_layers', type=int, default=model_params['num_layer'], help='Number of layers (including input and output)')
    parser.add_argument('--hidden_size', type=int, default=model_params['num_width'], help='Hidden layer size')
    
    # Multiple activation functions
    parser.add_argument('--activation', type=str, default=model_params['activation_func'], 
                       choices=['elu', 'relu', 'gelu', 'tanh', 'swish', 'leaky_relu'], help='Activation function')
    
    # Better optimizers and schedulers
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'cosine', 'step', 'exponential'], help='Learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model configuration:")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Activation: {args.activation}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"Physics weights - Weight1: {args.physics_weight1}, Weight2: {args.physics_weight2}")
    
    # Load and prepare data (use your existing function)
    train_loader, valid_loader = load_and_prepare_data(args.batch_size, args.num_samples)
    
    # Create model
    model = MLP(
        input_size=model_params['input_size'],
        hidden_size=args.hidden_size,
        output_size=model_params['output_size'],
        num_layers=args.num_layers,
        activation=args.activation
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        valid_loader,
        num_epochs=args.epochs,
        device=device,
        use_tensorboard=args.tensorboard,
        physics_weight1=args.physics_weight1,
        physics_weight2=args.physics_weight2,
        enable_grad=args.enable_grad_loss,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        learning_rate=args.learning_rate
    )
    
    print(f"Best validation loss: {min(val_losses):.6f}")
    print("Training setup completed!")


if __name__ == "__main__":
    main()