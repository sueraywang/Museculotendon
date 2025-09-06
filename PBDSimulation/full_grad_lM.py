import numpy as np
import matplotlib.pyplot as plt
from Physics import *
import math

def muscle_force(l_MT, l_M, activation):
    """
    Compute muscle force using FD muscle.muscle_force() method.
    """

    muscle = Muscle(
        l_m_opt=params['lMopt'], l_t_slack=params['lTslack'], v_m_max=params['vMmax'], 
        alpha_m_opt=params['alphaMopt'], f_m_opt=params['fMopt'],
        beta=params['beta'], a_min=params['aMin'], 
        tau_a=params['tauA'], tau_d=params['tauD']
    )
    
    # Calculate pennation angle
    alpha_m = muscle.calc_pennation_angle(l_M)
    
    # Calculate tendon length
    l_T = l_MT - l_M * math.cos(alpha_m)
    
    # Ensure valid tendon length
    if l_T < params['lTslack']:
        l_T = params['lTslack']
        l_M = (l_MT - l_T) / math.cos(alpha_m)
        alpha_m = muscle.calc_pennation_angle(l_M)
    
    # Calculate muscle velocity using muscle.compute_vel()
    v_m = muscle.compute_vel(l_M, l_MT, activation, alpha_m)
    
    # Calculate muscle force using the muscle's method
    lMtilde = l_M/params['lMopt']
    vMtilde = v_m/(params['lMopt'] * params['vMmax'])
    afl = curves['AFL'].calc_value(lMtilde)
    pfl = curves['PFL'].calc_value(lMtilde)
    fv = curves['FV'].calc_value(vMtilde)
    
    f_muscle = activation * afl * fv + pfl + params['beta'] * vMtilde
    
    return f_muscle * np.cos(alpha_m)
        

def finite_difference_gradient_l_M(l_MT, l_M, activation, h=1e-6):
    """
    Compute df/dl_M using finite differences of muscle_force.
    """
    f_plus = muscle_force(l_MT, l_M + h, activation)
    f_minus = muscle_force(l_MT, l_M, activation)
    
    df_dl_M = (f_plus - f_minus) / h
    
    return df_dl_M

def NN_gradient_l_M(l_MT, l_M, activation, model):
    """
    Compute df_nn/dl_M for f_nn = -C * dC/dl_M_tilde.
    
    We need: d/dl_M[-C * dC/dl_M_tilde] = -[dC/dl_M * dC/dl_M_tilde + C * d(dC/dl_M_tilde)/dl_M]
    
    Since dC/dl_M_tilde = dC/dl_M * l_m_opt, we have:
    d(dC/dl_M_tilde)/dl_M = l_m_opt * d²C/dl_M²
    
    So: df_nn/dl_M = -[dC/dl_M * dC/dl_M * l_m_opt + C * l_m_opt * d²C/dl_M²]
                   = -l_m_opt * [dC/dl_M² + C * d²C/dl_M²]
    """
    
    l_MT_tensor = torch.tensor(l_MT, dtype=torch.float32).reshape(-1, 1)
    l_M_tensor = torch.tensor(l_M, dtype=torch.float32).reshape(-1, 1)  # Physical units
    act = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
    inputs = torch.cat([l_MT_tensor, l_M_tensor, act], dim=1).detach().requires_grad_(True)
    
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
    
    dC_dl_M = torch.autograd.grad(
        outputs=C,
        inputs=l_M_with_grad,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]
    
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
    nn_grad_l_M = -params['lMopt'] * (dC_dl_M * dC_dl_M + C * d2C_dl_M2).item()
        
    return nn_grad_l_M

def analytical_gradient_l_M(l_MT, l_M, activation):
    """
    Compute df/dl_M analytically using the complete muscle force model.
    This implements the full analytical derivative including velocity coupling.
    """
    muscle = Muscle(
        l_m_opt=params['lMopt'], l_t_slack=params['lTslack'], v_m_max=params['vMmax'], 
        alpha_m_opt=params['alphaMopt'], f_m_opt=params['fMopt'],
        beta=params['beta'], a_min=params['aMin'], 
        tau_a=params['tauA'], tau_d=params['tauD']
    )
    
    # Current state
    alpha_m = muscle.calc_pennation_angle(l_M)
    l_T = l_MT - l_M * math.cos(alpha_m)
    v_m = muscle.compute_vel(l_M, l_MT, activation, alpha_m)
    
    # Normalized quantities
    l_m_tilde = l_M / params['lMopt']
    l_t_tilde = l_T / params['lTslack']
    v_m_tilde = v_m / (params['lMopt'] * params['vMmax'])
    
    # curve values and derivatives
    res_afl = muscle.curve_afl.calc_val_deriv(l_m_tilde)
    res_pfl = muscle.curve_pfl.calc_val_deriv(l_m_tilde)
    res_fv = muscle.curve_fv.calc_val_deriv(v_m_tilde)
    afl = res_afl[0]
    dafl_dl_m_tilde = res_afl[1]
    pfl = res_pfl[0]
    dpfl_dl_m_tilde = res_pfl[1]
    fv = res_fv[0]
    dfv_dv_tilde = res_fv[1]
    dtfl_dl_t_tilde = muscle.curve_tfl.calc_derivative(l_t_tilde, 1)
    
    # Current muscle force components
    f_m = activation * afl * fv + pfl + params['beta'] * v_m_tilde
    cos_alpha_m = math.cos(alpha_m)
    sin_alpha_m = math.sin(alpha_m)
    
    # === STEP 1: Compute basic derivatives ===
    
    # 1. Derivative of normalized muscle length
    inv_lMopt = 1.0 / params['lMopt']
    
    # 2. Derivative of pennation angle
    dalpha_dl_M = -muscle.h / (l_M**2 * cos_alpha_m)
    
    # === STEP 2: Compute dv_tilde/dl_M from force balance ===
    
    # Coefficients for the velocity derivative equation:
    # A * dv_tilde/dl_M = B_RHS - B_LHS
    
    # A = sensitivity of muscle force to velocity changes
    A = (activation * afl * dfv_dv_tilde + params['beta']) * cos_alpha_m
    
    # B_LHS = muscle force changes NOT due to velocity
    B_LHS = ((activation * dafl_dl_m_tilde * inv_lMopt * fv + 
                dpfl_dl_m_tilde * inv_lMopt) * cos_alpha_m - 
                f_m * sin_alpha_m * dalpha_dl_M)
    
    # B_RHS = tendon force changes due to length changes
    B_RHS = dtfl_dl_t_tilde / params['lTslack'] * (l_M * sin_alpha_m * dalpha_dl_M - cos_alpha_m)
    
    # Solve for dv_tilde/dl_M

    dv_tilde_dl_M = (B_RHS - B_LHS) / A
    
    # === STEP 3: Compute complete muscle force derivative ===
    
    # df_m/dl_M including velocity coupling
    df_m_dl_M = activation * (dafl_dl_m_tilde * inv_lMopt * fv + afl * dfv_dv_tilde * dv_tilde_dl_M) + dpfl_dl_m_tilde * inv_lMopt + params['beta'] * dv_tilde_dl_M
    
    # Final force derivative (product rule for f_m * cos(alpha_m))
    df_dl_M = (df_m_dl_M * cos_alpha_m - 
                f_m * sin_alpha_m * dalpha_dl_M)

    return df_dl_M

def generate_test_samples(num_samples=100):
    """
    Generate random valid (l_MT, l_M, activation) samples for testing.
    """
    
    # Define reasonable ranges
    l_MT_min = params['lMopt'] * 0.5 + params['lTslack']  # Minimum total length
    l_MT_max = params['lMopt'] * 1.5 + params['lTslack'] * 1.2  # Maximum total length
    l_M_min = params['lMopt'] * 0.5   # Minimum muscle fiber length
    l_M_max = params['lMopt'] * 1.5   # Maximum muscle fiber length
    activation_min, activation_max = 0.01, 1.0
    
    l_MT = np.random.uniform(l_MT_min, l_MT_max, num_samples)
    l_M = np.random.uniform(l_M_min, l_M_max, num_samples)
    activation = np.random.uniform(activation_min, activation_max, num_samples)
    
    samples = np.column_stack([l_MT, l_M, activation])
    return samples

def compare_gradients(samples, model=None):
    """
    Compare gradients from FD, analytical, and NN methods for l_M.
    """
    results = []
    
    print("Comparing df/dl_M gradients:")
    print("Method 1: Finite differences of muscle_force")
    print("Method 2: Analytical derivative of muscle force model")
    if model is not None:
        print("Method 3: Neural network f_nn = -C * dC/dl_M_tilde")
    print("=" * 80)
    
    for i, (l_MT, l_M, activation) in enumerate(samples):
        try:
            # Method 1: Finite differences of FD muscle force
            grad_FD = finite_difference_gradient_l_M(l_MT, l_M, activation)
            
            # Method 2: Analytical derivative
            grad_analytical = analytical_gradient_l_M(l_MT, l_M, activation)
            
            # Methods 3 & 4: Neural network gradients (if model provided)
            if model is not None:
                grad_NN = NN_gradient_l_M(l_MT, l_M, activation, model)
            else:
                grad_NN = 0.0
            
            # Compare gradients
            error_FD = abs(grad_FD - grad_analytical)
            rel_error_FD = error_FD / (abs(grad_analytical) + 1e-12)
            
            if model is not None:
                error_NN = abs(grad_NN - grad_analytical)
                rel_error_NN = error_NN / (abs(grad_analytical) + 1e-12)
            else:
                error_NN = 0.0
                rel_error_NN = 0.0
            
            # Get force for reference
            force = muscle_force(l_MT, l_M, activation)
            
            results.append({
                'l_MT': l_MT,
                'l_M': l_M,
                'activation': activation,
                'force': force,
                'grad_FD': grad_FD,
                'grad_analytical': grad_analytical,
                'grad_NN': grad_NN,
                'error_FD': error_FD,
                'rel_error_FD': rel_error_FD,
                'error_NN': error_NN,
                'rel_error_NN': rel_error_NN,
            })
            
            if i < 5:  # Print first 10 results
                print(f"Sample {i+1}: l_MT={l_MT:.4f}, l_M={l_M:.4f}, act={activation:.2f}")
                print(f"  Force:                    {force:.6e}")
                print(f"  df/dl_M - FD:             {grad_FD:.6e}")
                print(f"  df/dl_M - Analytical:     {grad_analytical:.6e}")
                print(f"  FD Error:         {error_FD:.6e} (rel: {rel_error_FD:.6e})")
                if model is not None:
                    print(f"  df_nn/dl_M - NN: {grad_NN:.6e}")
                    print(f"  NN Error:      {error_NN:.6e} (rel: {rel_error_NN:.6e})")
                print()
                
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue
    
    return results

def plot_comparison(results):
    """
    Plot comparison results for l_M gradients using analytical as reference.
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data with new field names
    analytical_grads = [r['grad_analytical'] for r in results]
    FD_grads = [r['grad_FD'] for r in results]
    NN_grads = [r['grad_NN'] for r in results]
    
    errors_FD = [r['error_FD'] for r in results]
    rel_errors_FD = [r['rel_error_FD'] for r in results]
    errors_NN = [r['error_NN'] for r in results]
    rel_errors_NN = [r['rel_error_NN'] for r in results]
    
    forces = [r['force'] for r in results]
    l_M_values = [r['l_M'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    
    # Plot 1: FD and NN vs Analytical (analytical as reference)
    ax1.scatter(analytical_grads, FD_grads, alpha=0.7, color='blue', s=50, label='FD vs Analytical')
    ax1.scatter(analytical_grads, NN_grads, alpha=0.7, color='red', s=50, label='NN vs Analytical')
    
    # Perfect agreement line
    min_val, max_val = min(FD_grads), max(FD_grads)
    
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('Analytical Gradient (df/dl_M) [Reference]')
    ax1.set_ylabel('Other Method Gradients (df/dl_M)')
    ax1.set_title('Gradient Method Comparison vs Analytical')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Relative error distribution
    ax2.hist(rel_errors_NN, bins=20, alpha=0.6, color='red', edgecolor='darkred', label='NN vs Analytical')
    ax2.set_xlabel('Relative Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Relative Error Distribution (vs Analytical)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def check_force_velocity_regime(l_MT, l_M, activation):
    """
    Check what force-velocity regime a sample is in.
    Returns: (regime, v_m_tilde, dfv_dv_tilde)
    """
    try:
        muscle = Muscle(
            l_m_opt=params['lMopt'], l_t_slack=params['lTslack'], v_m_max=params['vMmax'], 
            alpha_m_opt=params['alphaMopt'], f_m_opt=params['fMopt'],
            beta=params['beta'], a_min=params['aMin'], 
            tau_a=params['tauA'], tau_d=params['tauD']
        )
        
        alpha_m = muscle.calc_pennation_angle(l_M)
        v_m = muscle.compute_vel(l_M, l_MT, activation, alpha_m)
        v_m_tilde = v_m / (params['lMopt'] * params['vMmax'])
        
        # Get force-velocity curve derivative
        res_fv = muscle.curve_fv.calc_val_deriv(v_m_tilde)
        dfv_dv_tilde = res_fv[1]
        
        # Classify regime
        if abs(dfv_dv_tilde) < 1e-6:
            regime = "SATURATED"
        elif v_m_tilde > 0.1:
            regime = "ECCENTRIC"  # Lengthening
        elif v_m_tilde < -0.1:
            regime = "CONCENTRIC"  # Shortening
        else:
            regime = "ISOMETRIC"  # Near zero velocity
            
        return regime, v_m_tilde, dfv_dv_tilde
        
    except:
        return "ERROR", 0.0, 0.0

def is_valid_non_saturated_state(l_MT, l_M, activation, 
                                min_dfv_threshold=0.01, 
                                max_velocity_magnitude=2.0):
    """
    Enhanced validity check that excludes force-velocity saturation.
    
    Parameters:
    - min_dfv_threshold: Minimum |dfv_dv_tilde| to avoid saturation
    - max_velocity_magnitude: Maximum |v_m_tilde| to stay in active FV curve
    """
    try:
        # Basic geometric validity
        muscle = Muscle(
            l_m_opt=params['lMopt'], l_t_slack=params['lTslack'], v_m_max=params['vMmax'], 
            alpha_m_opt=params['alphaMopt'], f_m_opt=params['fMopt'],
            beta=params['beta'], a_min=params['aMin'], 
            tau_a=params['tauA'], tau_d=params['tauD']
        )
        
        alpha_m = muscle.calc_pennation_angle(l_M)
        l_T = l_MT - l_M * math.cos(alpha_m)
        
        # Check tendon length bounds
        if not (l_T >= muscle.l_t_slack and l_T <= muscle.l_t_slack * 1.4):
            return False
        
        # Check muscle length bounds
        if not (l_M >= params['lMopt'] * 0.6 and l_M <= params['lMopt'] * 1.4):
            return False
            
        # Check force-velocity regime
        regime, v_m_tilde, dfv_dv_tilde = check_force_velocity_regime(l_MT, l_M, activation)
        
        # Reject saturated samples
        if regime == "SATURATED":
            return False
            
        # Reject extreme velocities
        if abs(v_m_tilde) > max_velocity_magnitude:
            return False
            
        # Require meaningful force-velocity derivative
        if abs(dfv_dv_tilde) < min_dfv_threshold:
            return False
            
        return True
        
    except:
        return False

def generate_active_fv_samples(num_samples=100, 
                              target_regimes=["CONCENTRIC", "ISOMETRIC", "ECCENTRIC"],
                              velocity_bounds=(-1.5, 1.5)):
    """
    Generate samples specifically in the active force-velocity regime.
    
    Parameters:
    - target_regimes: Which FV regimes to include
    - velocity_bounds: (min_v_tilde, max_v_tilde) to constrain velocity range
    """
    samples = []
    regime_counts = {regime: 0 for regime in target_regimes}
    
    print("Generating samples in active force-velocity regimes...")
    print(f"Target regimes: {target_regimes}")
    print(f"Velocity bounds: {velocity_bounds}")
    print("="*60)
    
    # Strategy 1: Systematic parameter space exploration
    samples_per_regime = num_samples // len(target_regimes)
    
    for target_regime in target_regimes:
        print(f"\nGenerating {samples_per_regime} samples for {target_regime} regime:")
        
        regime_samples = 0
        attempts = 0
        max_attempts = samples_per_regime * 100
        
        while regime_samples < samples_per_regime and attempts < max_attempts:
            # Generate sample based on target regime
            if target_regime == "CONCENTRIC":
                # Favor conditions that produce shortening velocities
                l_MT = np.random.uniform(params['lMopt'] * 0.7 + params['lTslack'] * 0.9, 
                                       params['lMopt'] * 1.0 + params['lTslack'] * 1.05)
                l_M = np.random.uniform(params['lMopt'] * 0.8, params['lMopt'] * 1.1)
                activation = np.random.uniform(0.3, 0.9)
                
            elif target_regime == "ISOMETRIC":
                # Favor conditions near equilibrium
                l_MT = np.random.uniform(params['lMopt'] * 0.85 + params['lTslack'] * 0.95, 
                                       params['lMopt'] * 1.15 + params['lTslack'] * 1.05)
                l_M = np.random.uniform(params['lMopt'] * 0.9, params['lMopt'] * 1.1)
                activation = np.random.uniform(0.2, 0.8)
                
            elif target_regime == "ECCENTRIC":
                # Favor conditions that produce controlled lengthening
                l_MT = np.random.uniform(params['lMopt'] * 0.9 + params['lTslack'] * 1.0, 
                                       params['lMopt'] * 1.2 + params['lTslack'] * 1.1)
                l_M = np.random.uniform(params['lMopt'] * 0.7, params['lMopt'] * 1.0)
                activation = np.random.uniform(0.4, 0.9)
            
            # Check if sample is valid and in desired regime
            if is_valid_non_saturated_state(l_MT, l_M, activation):
                regime, v_m_tilde, dfv_dv_tilde = check_force_velocity_regime(l_MT, l_M, activation)
                
                # Check if in target regime and velocity bounds
                if (regime == target_regime and 
                    velocity_bounds[0] <= v_m_tilde <= velocity_bounds[1]):
                    
                    samples.append((l_MT, l_M, activation))
                    regime_counts[regime] += 1
                    regime_samples += 1
                    
                    if regime_samples <= 3:  # Show first few samples
                        print(f"  Sample {regime_samples}: v_tilde={v_m_tilde:.3f}, "
                              f"dfv_dv_tilde={dfv_dv_tilde:.3f}")
            
            attempts += 1
        
        print(f"  Generated {regime_samples} samples from {attempts} attempts")
    
    # Fill remaining with general active samples
    remaining = num_samples - len(samples)
    if remaining > 0:
        print(f"\nGenerating {remaining} additional active samples...")
        attempts = 0
        max_attempts = remaining * 200
        
        while len(samples) < num_samples and attempts < max_attempts:
            l_MT = np.random.uniform(params['lMopt'] * 0.7 + params['lTslack'] * 0.9, 
                                   params['lMopt'] * 1.3 + params['lTslack'] * 1.1)
            l_M = np.random.uniform(params['lMopt'] * 0.7, params['lMopt'] * 1.3)
            activation = np.random.uniform(0.1, 0.9)
            
            if is_valid_non_saturated_state(l_MT, l_M, activation):
                regime, v_m_tilde, dfv_dv_tilde = check_force_velocity_regime(l_MT, l_M, activation)
                
                if (regime in target_regimes and 
                    velocity_bounds[0] <= v_m_tilde <= velocity_bounds[1]):
                    
                    samples.append((l_MT, l_M, activation))
                    regime_counts[regime] += 1
            
            attempts += 1
    
    print(f"\nFINAL SAMPLE DISTRIBUTION:")
    print(f"Total samples: {len(samples)}")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} samples")
    
    return samples
    
# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED MUSCLE FORCE GRADIENT COMPARISON: df/dl_M")
    print("Comparing FD vs Analytical vs Neural Network Methods")
    print("=" * 80)
    
    try:
        # Try to load the neural network model
        model = None
        try:
            model = MLP(input_size=3, 
                       hidden_size=256, 
                       output_size=1, 
                       num_layers=5, 
                       activation='tanh')
            
            checkpoint = torch.load('TrainedModels/Muscles/test.pth', 
                                   map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print("Neural network model loaded successfully!")
        except Exception as e:
            print(f"Could not load neural network model: {e}")
            model = None
        
        print()
        
        # Generate test samples
        #samples = generate_test_samples(100)
        #"""
        samples = generate_active_fv_samples(
            num_samples=100,
            target_regimes=["CONCENTRIC", "ISOMETRIC", "ECCENTRIC"],  # Avoid extreme eccentric
            velocity_bounds=(-1.5, 1.5)  # Controlled velocity range
        )
        #"""
        
        # Compare gradients
        results = compare_gradients(samples, model)
        
        # Plot results
        plot_comparison(results)
        
        # Summary statistics
        if results:
            errors_FD = [r['error_FD'] for r in results]
            rel_errors_FD = [r['rel_error_FD'] for r in results]
            
            print("\nSUMMARY STATISTICS:")
            print(f"Number of samples: {len(results)}")
            print("\nFD vs Analytical:")
            print(f"  Mean absolute error: {np.mean(errors_FD):.2e}")
            print(f"  Max absolute error:  {np.max(errors_FD):.2e}")
            print(f"  Mean relative error: {np.mean(rel_errors_FD):.2e}")
            print(f"  Max relative error:  {np.max(rel_errors_FD):.2e}")
            print(f"  Std relative error:  {np.std(rel_errors_FD):.2e}")
            print(f"  Samples with rel error < 1%: {sum(1 for e in rel_errors_FD if e < 0.01)}/{len(rel_errors_FD)}")
            print(f"  Samples with rel error < 0.1%: {sum(1 for e in rel_errors_FD if e < 0.001)}/{len(rel_errors_FD)}")
            
            if model is not None:
                errors_NN = [r['error_NN'] for r in results]
                rel_errors_NN = [r['rel_error_NN'] for r in results]
                
                print("\nNN vs Analytical:")
                print(f"  Mean absolute error: {np.mean(errors_NN):.2e}")
                print(f"  Max absolute error:  {np.max(errors_NN):.2e}")
                print(f"  Mean relative error: {np.mean(rel_errors_NN):.2e}")
                print(f"  Max relative error:  {np.max(rel_errors_NN):.2e}")
                print(f"  Std relative error:  {np.std(rel_errors_NN):.2e}")
                print(f"  Samples with rel error < 1%: {sum(1 for e in rel_errors_NN if e < 0.01)}/{len(rel_errors_NN)}")
                print(f"  Samples with rel error < 0.1%: {sum(1 for e in rel_errors_NN if e < 0.001)}/{len(rel_errors_NN)}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()