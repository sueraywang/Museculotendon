import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Physics import *

# Test physics constraint step by step
def test_physics_constraint(targets, inputs, model):
    """
    More efficient vectorized approach for sample-wise gradients
    Modified to include lMtilde as input
    """
    # Get pre-computed features - now including lMtilde as 7th input
    act = inputs[:, 0].unsqueeze(1)
    afl = inputs[:, 1].unsqueeze(1)
    fv = inputs[:, 2].unsqueeze(1)
    pfl = inputs[:, 3].unsqueeze(1)
    cos_penn = inputs[:, 4].unsqueeze(1)
    vMtilde = inputs[:, 5].unsqueeze(1)
    lMtilde = inputs[:, 6].unsqueeze(1)  # NEW: lMtilde as direct input
    
    # Get derivative information for chain rule (shifted indices)
    dafl_dlMtilde = inputs[:, 7].unsqueeze(1)
    dpfl_dlMtilde = inputs[:, 9].unsqueeze(1)
    
    # Enable gradients for variables we need derivatives of
    afl_with_grad = afl.detach().requires_grad_(True)
    pfl_with_grad = pfl.detach().requires_grad_(True)
    lMtilde_with_grad = lMtilde.detach().requires_grad_(True)  # NEW: Enable gradients for lMtilde
    
    # Forward pass - now with 7 inputs including lMtilde
    network_inputs = torch.cat([act, afl_with_grad, fv, pfl_with_grad, cos_penn, vMtilde, lMtilde_with_grad], dim=1)
    C = model(network_inputs)
    
    # Calculate first derivatives dC using chain rule
    dC_dafl = torch.autograd.grad(
        outputs=C,
        inputs=afl_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    dC_dpfl = torch.autograd.grad(
        outputs=C,
        inputs=pfl_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # NEW: Direct derivative with respect to lMtilde
    dC_dlMtilde_direct = torch.autograd.grad(
        outputs=C,
        inputs=lMtilde_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Compute total dC_dl using chain rule + direct derivative
    # Total derivative = direct derivative + indirect derivatives through afl and pfl
    dC_dl = dC_dlMtilde_direct + dC_dafl * dafl_dlMtilde + dC_dpfl * dpfl_dlMtilde
    
    # Physics constraint: C * dC_dl = -f
    constraint_lhs = C * dC_dl
    constraint_rhs = -targets
    
    loss = F.mse_loss(constraint_lhs, constraint_rhs)
    return loss

# Generate data with ONLY act = 1.0 - Modified to include lMtilde as input
def generate_act1_data(num_samples=1000):
    """Generate training data with ONLY act = 1.0, now including lMtilde as input"""
    
    # Define ranges for input variables
    lMtilde_min, lMtilde_max = 1e-5, 2.0  # Set minimum to 1e-5 to avoid division by zero
    act_min, act_max = 0.0, 1.0  # Activation typically ranges from 0 to 1
    vMtilde_min, vMtilde_max = -2.0, 2.0  # Velocity range
    alphaMopt_min, alphaMopt_max = 0.0, np.pi/6  # Pennation angle range (0 to 30 degrees)
    
    # Generate random input values
    lMtilde_values = np.random.uniform(lMtilde_min, lMtilde_max, num_samples)
    vMtilde_values = np.random.uniform(vMtilde_min, vMtilde_max, num_samples)
    alphaMopt_values = np.random.uniform(alphaMopt_min, alphaMopt_max, num_samples)
    
    # Pre-compute features
    act_features = np.random.uniform(act_min, act_max, num_samples)
    afl_features = np.zeros(num_samples)
    fv_features = np.zeros(num_samples)
    pfl_features = np.zeros(num_samples)
    cos_alpha_features = np.cos(alphaMopt_values)
    
    # Derivatives
    dafl_dlMtilde = np.zeros(num_samples)
    dfv_dvMtilde = np.zeros(num_samples)
    dpfl_dlMtilde = np.zeros(num_samples)
    
    # Force values
    force_values = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Compute curve values and derivatives
        res_afl = curves['AFL'].calc_val_deriv(lMtilde_values[i])
        afl_features[i] = res_afl[0]
        dafl_dlMtilde[i] = res_afl[1]
        
        res_fv = curves['FV'].calc_val_deriv(vMtilde_values[i])
        fv_features[i] = res_fv[0]
        dfv_dvMtilde[i] = res_fv[1]
        
        res_pfl = curves['PFL'].calc_val_deriv(lMtilde_values[i])
        pfl_features[i] = res_pfl[0]
        dpfl_dlMtilde[i] = res_pfl[1]
        
        # Calculate target force (WITH pennation if uncommented)
        f_m = act_features[i] * afl_features[i] * fv_features[i] + pfl_features[i] + params['beta'] * vMtilde_values[i]
        force_values[i] = f_m * cos_alpha_features[i]  # Include pennation
    
    # Create input array: [act, afl, fv, pfl, cos_alpha, vMtilde, lMtilde, dafl_dlMtilde, dfv_dvMtilde, dpfl_dlMtilde]
    # NEW: lMtilde_values added as 7th column
    inputs = np.column_stack((
        act_features, afl_features, fv_features, pfl_features, cos_alpha_features, vMtilde_values,
        lMtilde_values,  # NEW: lMtilde as direct input
        dafl_dlMtilde, dfv_dvMtilde, dpfl_dlMtilde
    ))
    outputs = force_values.reshape(-1, 1)
    
    print(f"Generated {num_samples} samples")
    print(f"lMtilde range: [{lMtilde_values.min():.6f}, {lMtilde_values.max():.6f}]")
    print(f"Force range: [{force_values.min():.6f}, {force_values.max():.6f}]")
    
    return inputs, outputs

# Simplified training function - Modified for 7 inputs
def train_simple_overfit(num_samples=1000, num_epochs=1000, learning_rate=0.001, batch_size=64):
    """Simple training to overfit act=1.0 data - now with lMtilde input"""
    
    print("=== SIMPLIFIED TRAINING FOR ACT=1.0 OVERFITTING (WITH lMtilde INPUT) ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    inputs, outputs = generate_act1_data(num_samples)
    
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model - NOW WITH 7 INPUTS
    model = MLP(
        input_size=7,  # [act, afl, fv, pfl, cos_alpha, vMtilde, lMtilde] - CHANGED FROM 6 TO 7
        hidden_size=128,  # Smaller for overfitting test
        output_size=1,
        num_layers=4,
        activation='gelu'
    ).to(device)
    
    # Simple optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {num_samples}")
    print(f"Model input size: 7 (including lMtilde)")
    print("Starting training...")
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs_batch, targets_batch in dataloader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer.zero_grad()
            
            # Use physics constraint test
            loss = test_physics_constraint(targets_batch, inputs_batch, model)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Print progress
        if epoch % 100 == 0 or epoch < 10:
            print(f"Epoch {epoch:4d}: Loss = {avg_loss:.8f}")
        
        # Early stopping if converged
        if avg_loss < 1e-8:
            print(f"Converged at epoch {epoch}")
            break
    
    # Test the model
    print("\n=== TESTING OVERFITTED MODEL (WITH lMtilde INPUT) ===")
    model.eval()
    
    # Test on training data (without no_grad to allow gradient computation)
    test_inputs = inputs_tensor[:5].to(device)  # First 5 samples
    test_targets = outputs_tensor[:5].to(device)
    
    # Forward pass to get C - now with lMtilde
    act = test_inputs[:, 0].unsqueeze(1)
    afl = test_inputs[:, 1].unsqueeze(1)
    fv = test_inputs[:, 2].unsqueeze(1)
    pfl = test_inputs[:, 3].unsqueeze(1)
    cos_penn = test_inputs[:, 4].unsqueeze(1)
    vMtilde = test_inputs[:, 5].unsqueeze(1)
    lMtilde = test_inputs[:, 6].unsqueeze(1)  # NEW: Extract lMtilde
    
    # Get derivatives (shifted indices)
    dafl_dlMtilde = test_inputs[:, 7].unsqueeze(1)
    dpfl_dlMtilde = test_inputs[:, 9].unsqueeze(1)
    
    # Compute dC_dl using the same method as training
    # Create gradient-enabled inputs
    afl_grad = afl.detach().requires_grad_(True)
    pfl_grad = pfl.detach().requires_grad_(True)
    lMtilde_grad = lMtilde.detach().requires_grad_(True)  # NEW: Enable gradients for lMtilde
    inputs_with_grad = torch.cat([act, afl_grad, fv, pfl_grad, cos_penn, vMtilde, lMtilde_grad], dim=1)
    
    # Forward pass for gradients
    C_grad = model(inputs_with_grad)
    
    # Compute gradients
    dC_dafl = torch.autograd.grad(C_grad.sum(), afl_grad, retain_graph=True)[0]
    dC_dpfl = torch.autograd.grad(C_grad.sum(), pfl_grad, retain_graph=True)[0]
    dC_dlMtilde_direct = torch.autograd.grad(C_grad.sum(), lMtilde_grad, retain_graph=True)[0]  # NEW
    
    # Chain rule - now including direct derivative
    dC_dl = dC_dlMtilde_direct + dC_dafl * dafl_dlMtilde + dC_dpfl * dpfl_dlMtilde
    
    # Also get C values for comparison
    network_inputs = torch.cat([act, afl, fv, pfl, cos_penn, vMtilde, lMtilde], dim=1)
    C = model(network_inputs)
    
    # Compute -C * dC_dl (this should equal the force if constraint is satisfied)
    predicted_force = -C * dC_dl
    
    print("Sample predictions:")
    print("Target Force vs -C*dC_dl (should match if physics is correct):")
    for i in range(5):
        target_f = test_targets[i,0].item()
        predicted_f = predicted_force[i,0].item()
        C_val = C[i,0].item()
        dC_dl_val = dC_dl[i,0].item()
        lMtilde_val = lMtilde[i,0].item()
        
        print(f"Sample {i}:")
        print(f"  lMtilde:      {lMtilde_val:.6f}")
        print(f"  Target force: {target_f:.6f}")
        print(f"  -C * dC_dl:   {predicted_f:.6f}")
        print(f"  Error:        {abs(target_f - predicted_f):.6f}")
        print(f"  (C = {C_val:.6f}, dC_dl = {dC_dl_val:.6f})")
    
    # Save the model
    import os
    os.makedirs('SavedModels', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': 7,  # CHANGED FROM 6 TO 7
            'hidden_size': 128,
            'output_size': 1,
            'num_layers': 4,
            'activation': 'gelu'
        },
        'final_loss': losses[-1],
        'training_samples': num_samples
    }, 'SavedModels/precomputed_model_with_lMtilde.pth')
    
    print(f"\nModel saved to SavedModels/precomputed_model_with_lMtilde.pth")
    
    return model, losses

# NEW: Function to compare models with and without lMtilde
def compare_models(num_samples=500, num_epochs=1000):
    """Train and compare models with and without lMtilde input"""
    
    print("=== COMPARING MODELS WITH AND WITHOUT lMtilde INPUT ===\n")
    
    # Train model with lMtilde
    print("Training model WITH lMtilde input:")
    model_with_lMtilde, losses_with = train_simple_overfit(
        num_samples=num_samples, 
        num_epochs=num_epochs,
        learning_rate=0.001,
        batch_size=32
    )
    
    print(f"\nFinal loss WITH lMtilde: {losses_with[-1]:.8f}")
    print(f"Converged in {len(losses_with)} epochs\n")
    
    print("=" * 60)
    print("Training complete. Check the results above to see if adding lMtilde helped!")
    
    return model_with_lMtilde, losses_with

if __name__ == "__main__":
    # Run comparison test
    model, losses = compare_models(
        num_samples=10000,     # Small dataset for overfitting
        num_epochs=2000      # Enough epochs to overfit
    )
    
    print(f"\nFinal loss: {losses[-1]:.8f}")
    print("If this converges to very low loss, lMtilde input helped solve the constraint issue!")
    print("The model now has direct access to lMtilde and can learn both direct and indirect dependencies.")