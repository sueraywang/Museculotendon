import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import random
import time
import datetime
import os
import argparse
import math
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Physics import *

def muscle_force(l_MT, l_M, activation):
    """
    Compute muscle force using FD muscle.muscle_force() method.
    """
    
    # Calculate pennation angle
    alpha_m = Muscle.calc_pennation_angle(l_M, params)
    
    # Calculate tendon length
    l_T = l_MT - l_M * math.cos(alpha_m)
    
    # Ensure valid tendon length
    if l_T < params.lTslack:
        l_T = params.lTslack
        l_M = (l_MT - l_T) / math.cos(alpha_m)
        alpha_m = Muscle.calc_pennation_angle(l_M, params)
    
    # Calculate muscle velocity using muscle.compute_vel()
    v_m = Muscle.compute_vel(l_M, l_MT, activation, alpha_m, params)
    
    # Calculate muscle force using the muscle's method
    lMtilde = l_M/params.lMopt
    vMtilde = v_m/(params.lMopt * params.vMmax)
    afl = params.curve_afl.calc_value(lMtilde)
    pfl = params.curve_pfl.calc_value(lMtilde)
    fv = params.curve_fv.calc_value(vMtilde)
    
    f_muscle = activation * afl * fv + pfl + params.beta * vMtilde
    
    return f_muscle * np.cos(alpha_m)
        

def finite_difference_gradient_l_M(l_MT, l_M, activation, h=1e-6):
    """
    Compute df/dl_M using finite differences of muscle_force.
    """
    f_plus = muscle_force(l_MT, l_M + h, activation)
    f_minus = muscle_force(l_MT, l_M, activation)
    
    df_dl_M = (f_plus - f_minus) / h
    
    return df_dl_M

def physics_loss(inputs, forces, grad_l_M_targets, model, 
                 use_gradient_loss=False, gradient_loss_weight_l_M=1.0):
    
    # Get input components - following exact pattern from TrainMuscleWithPenn.py
    l_MT = inputs[:, 0].unsqueeze(1)
    l_M = inputs[:, 1].unsqueeze(1)
    activation = inputs[:, 2].unsqueeze(1)
    
    # Create inputs that require gradients - only for the variables we need gradients for
    l_MT_with_grad = l_MT.detach().requires_grad_(True)
    l_M_with_grad = l_M.detach().requires_grad_(True)
    
    # Reconstruct input tensor with gradient-enabled variables
    new_inputs = torch.cat([l_MT_with_grad, l_M_with_grad, activation], dim=1)
    
    # Forward pass to get C
    C = model(new_inputs)
    
    # Calculate first derivatives - following exact retain_graph pattern
    dC_dl_MT = torch.autograd.grad(
        outputs=C,
        inputs=l_MT_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    dC_dl_M = torch.autograd.grad(
        outputs=C,
        inputs=l_M_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Convert dC/dl_M to dC/dl_M_tilde using chain rule:
    dC_dl_M_tilde = dC_dl_M * params.lMopt
    
    # PRIMARY LOSS: f + C * dC/dl_M_tilde = 0
    physics_pred_primary = C * dC_dl_M_tilde
    loss_primary = F.mse_loss(physics_pred_primary, -forces)
    
    if use_gradient_loss:
        # Calculate second derivatives - following exact retain_graph pattern
        d2C_dl_M2 = torch.autograd.grad(
            outputs=dC_dl_M,
            inputs=l_M_with_grad,
            grad_outputs=torch.ones_like(dC_dl_M),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute NN gradients using constraint formulation
        # df_nn/dl_M = -l_m_opt * [dC/dl_M² + C * d²C/dl_M²]
        nn_grad_l_M = -params.lMopt * (dC_dl_M * dC_dl_M + C * d2C_dl_M2)
        
        # SECONDARY LOSS: df/dl_M = nn_grad_l_M
        loss_secondary = F.mse_loss(nn_grad_l_M, grad_l_M_targets)
        
        # Combine all terms - following exact pattern from TrainMuscleWithPenn.py
        total_loss = loss_primary + gradient_loss_weight_l_M * loss_secondary
    else:
        # Only primary loss
        total_loss = loss_primary
    
    return total_loss

def generate_training_data(num_samples=10000):
    """
    Generate training data by sampling random (l_MT, l_M, activation) combinations.
    Precompute forces and analytical gradients for efficiency.
    
    Returns:
        inputs: [l_MT, l_M, activation] - physical units
        forces: [muscle_force] - normalized
        grad_l_M: [df/dl_M] - normalized analytical gradients
    """
    
    # Define ranges for input variables
    l_MT_min = params.lMopt * 0.5 + params.lTslack  # Minimum total length
    l_MT_max = params.lMopt * 1.5 + params.lTslack * 1.2  # Maximum total length
    l_M_min = params.lMopt * 0.5   # Minimum muscle fiber length
    l_M_max = params.lMopt * 1.5   # Maximum muscle fiber length
    activation_min, activation_max = 0.0, 1.0
    
    # Arrays to store results
    force_values = []
    grad_l_M_values = []
    
    print("Generating training data with precomputed gradients...")
        
    l_MT = np.random.uniform(l_MT_min, l_MT_max, num_samples)
    l_M = np.random.uniform(l_M_min, l_M_max, num_samples)
    activation = np.random.uniform(activation_min, activation_max, num_samples)
    for i in range(num_samples):
        force_values.append(muscle_force(l_MT[i], l_M[i], activation[i]))
        grad_l_M_values.append(finite_difference_gradient_l_M(l_MT[i], l_M[i], activation[i]))
    
    # Convert to numpy arrays
    inputs = np.column_stack([l_MT, l_M, activation])
    forces = np.array(force_values).reshape(-1, 1)
    grad_l_M = np.array(grad_l_M_values).reshape(-1, 1)
    
    print(f"Input ranges:")
    print(f"  l_MT: [{inputs[:, 0].min():.3f}, {inputs[:, 0].max():.3f}]")
    print(f"  l_M: [{inputs[:, 1].min():.3f}, {inputs[:, 1].max():.3f}]")
    print(f"  activation: [{inputs[:, 2].min():.3f}, {inputs[:, 2].max():.3f}]")
    print(f"Force range (normalized): [{forces.min():.3f}, {forces.max():.3f}]")
    print(f"Gradient l_M range (normalized): [{grad_l_M.min():.3f}, {grad_l_M.max():.3f}]")
    
    np.savez('full_model_data.npz', array1=inputs, array2=forces, array3=grad_l_M)
    
    return inputs, forces, grad_l_M

def train_model(model, train_loader, valid_loader, num_epochs=100, device='cuda', 
                use_tensorboard=False, optimizer_type='adamw', scheduler_type='plateau', learning_rate=0.001,
                use_gradient_loss=False, gradient_loss_weight_l_M=1.0):
    """Training function for the physics-based model"""
    
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
        log_dir = os.path.join('TrainedResults/Muscles/lMT_lM_ACT', runTime)
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
        
        for inputs, forces, grad_l_M_targets in train_loader:
            # Move data to GPU
            inputs = inputs.to(device)
            forces = forces.to(device)
            grad_l_M_targets = grad_l_M_targets.to(device)
            
            optimizer.zero_grad()
            
            # Use physics-based loss function
            loss = physics_loss(inputs, forces, grad_l_M_targets, model,
                               use_gradient_loss=use_gradient_loss,
                               gradient_loss_weight_l_M=gradient_loss_weight_l_M)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            pbar.update(1)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, forces, grad_l_M_targets in valid_loader:
                # Move data to GPU
                inputs = inputs.to(device)
                forces = forces.to(device)
                grad_l_M_targets = grad_l_M_targets.to(device)
                
                # During validation, use the same physics loss
                with torch.enable_grad():
                    # Don't detach during validation - follow reference pattern exactly
                    loss = physics_loss(inputs, forces, grad_l_M_targets, model,
                                       use_gradient_loss=use_gradient_loss,
                                       gradient_loss_weight_l_M=gradient_loss_weight_l_M)
                
                val_loss += loss.item()
        
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
            }, os.path.join('TrainedModels/Muscles', 'lm_lmt_model.pth'))
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

def load_and_prepare_data(batch_size, num_samples=50000):
    """Load and prepare physics-based training data with precomputed gradients"""
    
    # Generate training data with precomputed gradients
    inputs, forces, grad_l_M= generate_training_data(num_samples)
    """
    loaded_data = np.load('full_model_data.npz')
    # Access individual arrays by their names
    inputs = loaded_data['array1']
    forces = loaded_data['array2']
    grad_l_M = loaded_data['array3']
    """
    
    # Convert to tensor format
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    forces_tensor = torch.tensor(forces, dtype=torch.float32)
    grad_l_M_tensor = torch.tensor(grad_l_M, dtype=torch.float32)
    
    # Create dataset with four components: inputs, forces, grad_l_M
    dataset = TensorDataset(inputs_tensor, forces_tensor, grad_l_M_tensor)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
    
    return train_loader, valid_loader

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train physics-constraint muscle model with gradient losses')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of training samples to generate')
    
    # Model configuration
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--activation', type=str, default='tanh', 
                       choices=['elu', 'relu', 'gelu', 'tanh', 'swish', 'leaky_relu'], help='Activation function')
    
    # Training configuration
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'step', 'exponential'], help='Learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Gradient loss configuration
    parser.add_argument('--use_gradient_loss', action='store_true', default=False, 
                       help='Enable gradient loss terms for improved gradient accuracy')
    parser.add_argument('--gradient_loss_weight_l_M', type=float, default=1.0, 
                       help='Weight for l_M gradient loss term')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model configuration:")
    print(f"  - Input size: 3 (l_MT, l_M, activation)")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Activation: {args.activation}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Use gradient loss: {args.use_gradient_loss}")
    if args.use_gradient_loss:
        print(f"  - Gradient loss weight (l_M): {args.gradient_loss_weight_l_M}")
    
    # Load and prepare data
    train_loader, valid_loader = load_and_prepare_data(args.batch_size, args.num_samples)
    
    # Create model
    model = MLP(
        input_size=3,  # l_MT, l_M, activation
        hidden_size=args.hidden_size,
        output_size=1,  # constraint C
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
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        learning_rate=args.learning_rate,
        use_gradient_loss=args.use_gradient_loss,
        gradient_loss_weight_l_M=args.gradient_loss_weight_l_M
    )
    
    print(f"Best validation loss: {min(val_losses):.6f}")
    print("Training completed!")

if __name__ == "__main__":
    main()