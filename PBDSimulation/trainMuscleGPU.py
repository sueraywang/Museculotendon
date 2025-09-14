import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import time
import datetime
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Physics import *

class MuscleModel:
    def __init__(self, use_pennation=False, use_damping=False):
        self.use_pennation = use_pennation
        self.use_damping = use_damping
        self.input_dim = 4 if use_pennation else 3
    
    def muscle_force(self, lMtilde: float, act: float, vMtilde: float, alphaMopt: float = None) -> float:
        """Calculate muscle force with optional pennation angle or damping"""
        afl = params.curve_afl.calc_value(lMtilde)
        pfl = params.curve_pfl.calc_value(lMtilde)
        fv = params.curve_fv.calc_value(vMtilde)
        
        if self.use_pennation and alphaMopt is not None:
            # Full muscle force model
            penn = np.arcsin(np.sin(alphaMopt)/lMtilde)
            f_m = act * afl * fv + pfl + params.beta * vMtilde
            return f_m * np.cos(penn)
        elif self.use_damping:
            # Without pennation angle but with damping
            return act * afl * fv + pfl + params.beta * vMtilde
        else:
            # Without pennation angle nor damping
            return act * afl * fv + pfl
    
    def muscle_force_derivative(self, lMtilde: float, act: float, vMtilde: float, alphaMopt: float = None) -> np.ndarray:
        """Calculate analytical derivatives of muscle force with optional pennation angle or damping"""
        # Clamp lMtilde to minimum value to avoid division by zero
        lMtilde = max(lMtilde, 1e-5)
        
        # Get values and derivatives from curves
        res_afl = params.curve_afl.calc_val_deriv(lMtilde)
        afl = res_afl[0]
        afl_deriv = res_afl[1]
        
        res_fv = params.curve_fv.calc_val_deriv(vMtilde)
        fv = res_fv[0]
        fv_deriv = res_fv[1]
        
        if self.use_pennation and alphaMopt is not None:
            # With pennation angle
            res_pfl = params.curve_pfl.calc_val_deriv(lMtilde)
            pfl = res_pfl[0]
            pfl_deriv = res_pfl[1]
            
            # Compute pennation angle and its derivatives
            sin_alpha = np.sin(alphaMopt)
            penn = np.arcsin(sin_alpha / lMtilde)
            cos_penn = np.cos(penn)
            sin_penn = np.sin(penn)
            
            # Compute f_m (muscle force without pennation)
            f_m = act * afl * fv + pfl + params.beta * vMtilde
            
            # Partial derivatives of f_m
            df_m_dlMtilde = act * afl_deriv * fv + pfl_deriv
            df_m_dvMtilde = act * afl * fv_deriv + params.beta
            
            # Partial derivative of pennation angle w.r.t lMtilde
            dpenn_dlMtilde = -sin_alpha / (lMtilde**2 * np.sqrt(1 - (sin_alpha/lMtilde)**2))
            
            # Compute total partial derivatives using product rule
            dF_dlMtilde = df_m_dlMtilde * cos_penn - f_m * sin_penn * dpenn_dlMtilde
            dF_dvMtilde = df_m_dvMtilde * cos_penn
            
        else:
            pfl_deriv = params.curve_pfl.calc_derivative(lMtilde, 1)
            dF_dlMtilde = act * afl_deriv * fv + pfl_deriv

            if self.use_damping:
                # Without pennation angle but with damping
                dF_dvMtilde = act * afl * fv_deriv + params.beta
            else:
                # Without pennation angle nor damping
                dF_dvMtilde = act * afl * fv_deriv
        
        return np.array([dF_dlMtilde, dF_dvMtilde])

def custom_physics_loss(targets, dF_dl_targets, dF_dv_targets, inputs, model, 
                       physics_weight1=1.0, physics_weight2=1.0, ENABLE_GRAD=False, use_pennation=False):
    """
    Loss function
    """
    # Get input components
    act = inputs[:, 1].unsqueeze(1)
    lMtilde = inputs[:, 0].unsqueeze(1)
    vMtilde = inputs[:, 2].unsqueeze(1)
    
    if use_pennation:
        alphaMopt = inputs[:, 3].unsqueeze(1)
    
    # Create inputs that require gradients for lMtilde and vMtilde only
    lMtilde_with_grad = lMtilde.detach().requires_grad_(True)
    vMtilde_with_grad = vMtilde.detach().requires_grad_(True)
    
    # Reconstruct input tensor with gradient-enabled variables
    if use_pennation:
        new_inputs = torch.cat([lMtilde_with_grad, act, vMtilde_with_grad, alphaMopt], dim=1)
    else:
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
    
    # LOSS OF FORCE PREDICTION: f + C * dC_dl = 0
    pred_force = C * dC_dl
    force_loss = F.mse_loss(pred_force, -targets)
    
    if ENABLE_GRAD:
        # LOSS OF GRADIENT PREDICTION WRT lMtilde: df/dl + (dC_dl * dC_dl + C * d2C_dldl) = 0
        pred_grad_lm = dC_dl * dC_dl + C * d2C_dldl
        grad_loss_lm = F.mse_loss(pred_grad_lm, -dF_dl_targets)
        
        # LOSS OF GRADIENT PREDICTION WRT vMtilde: df/dv + (dC_dv * dC_dl + C * d2C_dldv) = 0
        pred_grad_vm = dC_dv * dC_dl + C * d2C_dldv
        grad_loss_vm = F.mse_loss(pred_grad_vm, -dF_dv_targets)
        
        # Combine all terms
        total_loss = force_loss + physics_weight1 * grad_loss_lm + physics_weight2 * grad_loss_vm
    else:
        total_loss = force_loss
    
    return total_loss

def train_model(model, train_loader, valid_loader, num_epochs=100, early_stop_patience=200, device='cuda', 
                use_tensorboard=False, physics_weight1=1.0, physics_weight2=1.0, enable_grad=False,
                optimizer_type='adam', scheduler_type='plateau', learning_rate=0.001, use_pennation=False):
    """Training function with configurable pennation support"""
    
    model = model.to(device)
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Choose optimizer
    optimizers = {
        'adamw': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4),
        'adam': optim.Adam(model.parameters(), lr=learning_rate),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4),
        'rmsprop': optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    }
    optimizer = optimizers.get(optimizer_type)
    
    # Choose scheduler
    schedulers = {
        'plateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False),
        'step': optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8),
        'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    }
    scheduler = schedulers.get(scheduler_type)
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if use_tensorboard:
        model_type = 'fullForce' if use_pennation else 'noPenn'
        log_dir = os.path.join('TrainedResults/Muscles', model_type, runTime)
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
            
            loss = custom_physics_loss(targets, dF_dl_targets, dF_dv_targets, inputs, model, 
                                     physics_weight1, physics_weight2, enable_grad, use_pennation)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
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
                    loss = custom_physics_loss(targets_detached, dF_dl_targets_detached, dF_dv_targets_detached, 
                                             inputs_detached, model, physics_weight1, physics_weight2, enable_grad, use_pennation)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(valid_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        elif scheduler_type in ['step', 'exponential']:
            scheduler.step()
        
        # Log losses and timing to TensorBoard if enabled
        if use_tensorboard and writer is not None:
            writer.add_scalars('Losses/train_vs_validation', {
                'train': train_loss,
                'validation': val_loss
            }, epoch)
            writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
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
                'use_pennation': use_pennation,
                'model_config': {
                    'input_size': model.input_size,
                    'hidden_size': model.hidden_size,
                    'output_size': model.output_size,
                    'num_layers': model.num_layers,
                }
            }, os.path.join('TrainedModels/Muscles', model_params.get('model_name', 'a_test_model.pth')))
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

def generate_focused_data_method(muscle_model, num_samples=10000, use_pennation=False):
    """Generate synthetic data with configurable pennation and damping"""
    
    # Define ranges for input variables
    act_min, act_max = 0.0, 1.0
    lMtilde_min, lMtilde_max = 1e-5, 2.0
    vMtilde_min, vMtilde_max = -2.0, 2.0
    
    # Generate random input values
    act_values = np.random.uniform(act_min, act_max, num_samples)
    lMtilde_values = np.random.uniform(lMtilde_min, lMtilde_max, num_samples)
    
    if use_pennation:
        alphaMopt_min, alphaMopt_max = 0.0, np.pi/5
        alphaMopt_values = np.random.uniform(alphaMopt_min, alphaMopt_max, num_samples)
    
    # Focus around zero velocity (normal distribution)
    n1 = int(num_samples * 0.8)
    n2 = int(num_samples * 0.15)
    n3 = num_samples - n1 - n2
    
    # Generate and combine velocity values
    values1 = np.random.uniform(vMtilde_min, vMtilde_max, n1)
    values2 = np.random.uniform(-0.3, 0.3, n2)
    values3 = np.zeros(n3)
    vMtilde_values = np.concatenate([values1, values2, values3])
    
    # Calculate corresponding muscle forces and derivatives
    force_values = np.zeros(num_samples)
    dF_dl_values = np.zeros(num_samples)
    dF_dv_values = np.zeros(num_samples)
    
    for i in range(num_samples):
        if use_pennation:
            # Ensure sin(alphaMopt)/lMtilde <= 1 to avoid domain error in arcsin
            if np.sin(alphaMopt_values[i]) / lMtilde_values[i] > 1.0:
                lMtilde_values[i] = max(np.sin(alphaMopt_values[i]) / 0.99, 1e-5)
            
            force_values[i] = muscle_model.muscle_force(lMtilde_values[i], act_values[i], 
                                                       vMtilde_values[i], alphaMopt_values[i])
            derivatives = muscle_model.muscle_force_derivative(lMtilde_values[i], act_values[i], 
                                                              vMtilde_values[i], alphaMopt_values[i])
        else:
            force_values[i] = muscle_model.muscle_force(lMtilde_values[i], act_values[i], vMtilde_values[i])
            derivatives = muscle_model.muscle_force_derivative(lMtilde_values[i], act_values[i], vMtilde_values[i])
        
        dF_dl_values[i] = derivatives[0]  # dF/dlMtilde
        dF_dv_values[i] = derivatives[1]  # dF/dvMtilde
    
    # Create input and output arrays
    if use_pennation:
        inputs = np.column_stack((lMtilde_values, act_values, vMtilde_values, alphaMopt_values))
    else:
        inputs = np.column_stack((lMtilde_values, act_values, vMtilde_values))
    
    outputs = force_values.reshape(-1, 1)
    dF_dl_outputs = dF_dl_values.reshape(-1, 1)
    dF_dv_outputs = dF_dv_values.reshape(-1, 1)
    
    return inputs, outputs, dF_dl_outputs, dF_dv_outputs

def load_and_prepare_data(muscle_model, batch_size, num_samples=100000, use_pennation=False):
    """Load and prepare data with configurable pennation support"""
    # Generate synthetic data
    inputs, outputs, dF_dl_outputs, dF_dv_outputs = generate_focused_data_method(muscle_model, num_samples, use_pennation)
    
    input_dim = 4 if use_pennation else 3
    print(f"Generated data shape: {inputs.shape[0]} samples with {inputs.shape[1]} features")
    print(f"Expected input dimension: {input_dim}")
    print(f"Output shape: {outputs.shape}")
    print(f"dF/dl shape: {dF_dl_outputs.shape}")
    print(f"dF/dv shape: {dF_dv_outputs.shape}")
    
    # Verify input dimension matches expectation
    assert inputs.shape[1] == input_dim, f"Input dimension mismatch: got {inputs.shape[1]}, expected {input_dim}"
    
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
    global muscle_model
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train muscle force prediction model')
    parser.add_argument('--use_pennation', action='store_true', default=False, 
                       help='Include pennation angle in the muscle force model')
    parser.add_argument('--use_damping', action='store_true', default=False, 
                       help='Include damping term in the muscle force model')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of synthetic data samples to generate')
    parser.add_argument('--early_stop_patience', type=int, default=200, help='Early stop patience of training')
    parser.add_argument('--physics_weight1', type=float, default=1.0, help='Weight for first physics regularization term')
    parser.add_argument('--physics_weight2', type=float, default=1.0, help='Weight for second physics regularization term')
    parser.add_argument('--enable_grad_loss', action='store_true', default=False, 
                       help='Enable gradient minimization in loss function')
    
    # Configurable model parameters
    parser.add_argument('--num_layers', type=int, default=model_params.get('num_layer', 5), 
                       help='Number of layers (including input and output)')
    parser.add_argument('--hidden_size', type=int, default=model_params.get('num_width', 128), 
                       help='Hidden layer size')
    parser.add_argument('--activation', type=str, default=model_params.get('activation_func', 'elu'), 
                       choices=['elu', 'relu', 'gelu', 'tanh', 'swish', 'leaky_relu'], help='Activation function')
    
    # Optimizer and scheduler options
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'step', 'exponential'], help='Learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize muscle model based on pennation setting
    muscle_model = MuscleModel(use_pennation=args.use_pennation, use_damping=args.use_damping)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Force model includes Pennation: {'Yes' if args.use_pennation else 'No'}")
    print(f"Force model includes Damping: {'Yes' if args.use_damping else 'No'}")
    print(f"Model configuration:")
    print(f"  - Input size: {muscle_model.input_dim}")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Activation: {args.activation}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"Physics weights - Weight1: {args.physics_weight1}, Weight2: {args.physics_weight2}")
    print(f"Enable gradient loss: {args.enable_grad_loss}")
    
    # Load and prepare data
    train_loader, valid_loader = load_and_prepare_data(muscle_model, args.batch_size, args.num_samples, args.use_pennation)
    
    # Create model with configurable input size
    model = MLP(
        input_size=muscle_model.input_dim,
        hidden_size=args.hidden_size,
        output_size=model_params.get('output_size', 1),
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
        early_stop_patience = args.early_stop_patience,
        device=device,
        use_tensorboard=args.tensorboard,
        physics_weight1=args.physics_weight1,
        physics_weight2=args.physics_weight2,
        enable_grad=args.enable_grad_loss,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        learning_rate=args.learning_rate,
        use_pennation=args.use_pennation
    )
    
    print(f"Best validation loss: {min(val_losses):.6f}")
    print("Training completed!")


if __name__ == "__main__":
    main()