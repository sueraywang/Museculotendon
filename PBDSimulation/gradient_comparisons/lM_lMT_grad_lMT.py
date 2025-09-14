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
    
    return f_muscle

def finite_difference_gradient_l_MT(l_MT, l_M, activation, h=1e-6):
    """
    Compute df/dl_MT using finite differences of muscle_force.
    """
    f_plus = muscle_force(l_MT + h, l_M, activation)
    f_minus = muscle_force(l_MT - h, l_M, activation)
    
    df_dl_MT = (f_plus - f_minus) / (2 * h)
    
    return df_dl_MT

def NN_gradient_l_MT(l_MT, l_M, activation, model):
    """
    Compute df_nn/dl_MT for f_nn = -C * dC/dl_M_tilde.
    
    We need: d/dl_MT[-C * dC/dl_M_tilde] = -[dC/dl_MT * dC/dl_M_tilde + C * d(dC/dl_M_tilde)/dl_MT]
    
    Since dC/dl_M_tilde = dC/dl_M * l_m_opt, we have:
    d(dC/dl_M_tilde)/dl_MT = l_m_opt * d²C/dl_MT dl_M
    
    So: df_nn/dl_MT = -l_m_opt * [dC/dl_MT * dC/dl_M + C * d²C/dl_MT dl_M]
    """
    try:
        with torch.enable_grad():
            inputs = torch.tensor([[l_MT, l_M, activation]], 
                                  requires_grad=True, dtype=torch.float32)
            
            C = model(inputs)
            
            # First derivatives
            first_grads = torch.autograd.grad(C, inputs, 
                                             grad_outputs=torch.ones_like(C), 
                                             create_graph=True, 
                                             retain_graph=True)[0]
            
            dC_dl_MT = first_grads[0, 0]  # derivative w.r.t. l_MT (first input)
            dC_dl_M = first_grads[0, 1]   # derivative w.r.t. l_M (second input)
            
            # Second derivative d²C/dl_MT dl_M (mixed partial)
            d2C_dl_MT_dl_M = torch.autograd.grad(dC_dl_MT, inputs, 
                                                grad_outputs=torch.ones_like(dC_dl_MT), 
                                                create_graph=False, 
                                                retain_graph=False)[0][0, 1]
            
            # Extract values
            C_val = C.item()
            dC_dl_MT_val = dC_dl_MT.item()
            dC_dl_M_val = dC_dl_M.item()
            d2C_dl_MT_dl_M_val = d2C_dl_MT_dl_M.item()
        
        # Analytical derivative of f_nn = -C * dC/dl_M_tilde w.r.t. l_MT
        # df_nn/dl_MT = -l_m_opt * [dC/dl_MT * dC/dl_M + C * d²C/dl_MT dl_M]
        df_nn_dl_MT = -params['lMopt'] * (dC_dl_MT_val * dC_dl_M_val + C_val * d2C_dl_MT_dl_M_val)
        
        return df_nn_dl_MT
        
    except Exception as e:
        print(f"Error in NN gradient: {e}")
        return 0.0

def analytical_gradient_l_MT(l_MT, l_M, activation):
    """
    Compute df/dl_MT analytically using the complete muscle force model.
    
    Key insight: l_MT affects the system only through tendon length l_T,
    and ∂l_T/∂l_MT = 1, ∂alpha_m/∂l_MT = 0.
    
    This makes the l_MT gradient much simpler than the l_M gradient!
    """
    try:
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
        
        # Force-length and force-velocity relationships
        afl = muscle.curve_afl.calc_value(l_m_tilde)
        
        # Current velocity and force-velocity curve
        res_fv = muscle.curve_fv.calc_val_deriv(v_m_tilde)
        fv = res_fv[0]
        dfv_dv_tilde = res_fv[1]
        
        cos_alpha_m = math.cos(alpha_m)
        
        # === KEY INSIGHT: Simplified derivation for l_MT ===
        # Since ∂α_m/∂l_MT = 0 and ∂l_T/∂l_MT = 1:
        
        # 1. Derivative of tendon length
        dl_t_tilde_dl_MT = 1.0 / params['lTslack']
        
        # 2. Derivative of tendon force-length curve
        try:
            dtfl_dl_t_tilde = muscle.curve_tfl.calc_derivative(l_t_tilde, 1)
        except AttributeError:
            # Fallback to finite differences if derivative method doesn't exist
            h_small = 1e-8
            dtfl_dl_t_tilde = (muscle.curve_tfl.calc_value(l_t_tilde + h_small) - 
                              muscle.curve_tfl.calc_value(l_t_tilde - h_small)) / (2 * h_small)
        
        # 3. From force balance: f_m * cos(α_m) = TFL(l_t_tilde)
        # Differentiating: ∂f_m/∂l_MT * cos(α_m) = ∂TFL/∂l_t_tilde * ∂l_t_tilde/∂l_MT
        # Since ∂α_m/∂l_MT = 0, the sin(α_m) term vanishes!
        
        # 4. Since f_m = a*afl*fv + pfl + β*v_m_tilde, and only v_m_tilde depends on l_MT:
        # ∂f_m/∂l_MT = (a*afl*dfv_dv_tilde + β) * ∂v_m_tilde/∂l_MT
        
        # 5. From force balance derivative:
        # (a*afl*dfv_dv_tilde + β) * ∂v_m_tilde/∂l_MT * cos(α_m) = dtfl_dl_t_tilde * dl_t_tilde_dl_MT
        
        # 6. Solve for velocity derivative:
        denominator = (activation * afl * dfv_dv_tilde + params['beta']) * cos_alpha_m
        
        if abs(denominator) > 1e-12:
            dv_tilde_dl_MT = (dtfl_dl_t_tilde * dl_t_tilde_dl_MT) / denominator
        else:
            dv_tilde_dl_MT = 0.0
            print("Warning: Small denominator in l_MT velocity derivative calculation")
        
        # 7. Complete muscle force derivative
        # ∂f_m/∂l_MT = (a*afl*dfv_dv_tilde + β) * dv_tilde_dl_MT
        df_m_dl_MT = (activation * afl * dfv_dv_tilde + params['beta']) * dv_tilde_dl_MT
        
        # 8. Final force derivative (no pennation angle term since ∂α_m/∂l_MT = 0)
        df_dl_MT = df_m_dl_MT * cos_alpha_m
        
        return df_dl_MT
        
    except Exception as e:
        print(f"Error in analytical gradient: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

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
    Compare gradients from FD, analytical, and NN methods for l_MT.
    """
    results = []
    
    print("Comparing df/dl_MT gradients:")
    print("Method 1: Finite differences of muscle_force")
    print("Method 2: Analytical derivative of muscle force model")
    if model is not None:
        print("Method 3: Neural network f_nn = -C * dC/dl_M_tilde")
    print("=" * 80)
    
    for i, (l_MT, l_M, activation) in enumerate(samples):
        try:
            # Method 1: Finite differences of FD muscle force
            grad_FD = finite_difference_gradient_l_MT(l_MT, l_M, activation)
            
            # Method 2: Analytical derivative
            grad_analytical = analytical_gradient_l_MT(l_MT, l_M, activation)
            
            # Method 3: Neural network gradients (if model provided)
            if model is not None:
                grad_NN = NN_gradient_l_MT(l_MT, l_M, activation, model)
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
            
            if i < 5:  # Print first 5 results
                print(f"Sample {i+1}: l_MT={l_MT:.4f}, l_M={l_M:.4f}, act={activation:.2f}")
                print(f"  Force:                    {force:.6e}")
                print(f"  df/dl_MT - FD:            {grad_FD:.6e}")
                print(f"  df/dl_MT - Analytical:    {grad_analytical:.6e}")
                print(f"  FD Error:         {error_FD:.6e} (rel: {rel_error_FD:.6e})")
                if model is not None:
                    print(f"  df_nn/dl_MT - NN:  {grad_NN:.6e}")
                    print(f"  NN Error:       {error_NN:.6e} (rel: {rel_error_NN:.6e})")
                print()
                
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue
    
    return results

def plot_comparison(results):
    """
    Plot comparison results for l_MT gradients using analytical as reference.
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    analytical_grads = [r['grad_analytical'] for r in results]
    FD_grads = [r['grad_FD'] for r in results]
    NN_grads = [r['grad_NN'] for r in results]
    
    errors_FD = [r['error_FD'] for r in results]
    rel_errors_FD = [r['rel_error_FD'] for r in results]
    errors_NN = [r['error_NN'] for r in results]
    rel_errors_NN = [r['rel_error_NN'] for r in results]
    
    forces = [r['force'] for r in results]
    l_MT_values = [r['l_MT'] for r in results]
    
    # Check if we have NN data
    has_NN_data = any(r['grad_NN'] != 0.0 for r in results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    
    # Plot 1: FD and NN vs Analytical (analytical as reference)
    ax1.scatter(analytical_grads, FD_grads, alpha=0.7, color='blue', s=50, label='FD vs Analytical')
    if has_NN_data:
        ax1.scatter(analytical_grads, NN_grads, alpha=0.7, color='red', s=30, label='NN vs Analytical')
    
    # Perfect agreement line
    all_grads = analytical_grads + FD_grads
    if has_NN_data:
        all_grads += NN_grads
    min_val, max_val = min(all_grads), max(all_grads)
    
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('Analytical Gradient (df/dl_MT) [Reference]')
    ax1.set_ylabel('Other Method Gradients (df/dl_MT)')
    ax1.set_title('Gradient Method Comparison vs Analytical')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Relative error distribution
    ax2.hist(rel_errors_FD, bins=20, alpha=0.6, color='blue', edgecolor='darkblue', label='FD vs Analytical')
    if has_NN_data:
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
    print("ENHANCED MUSCLE FORCE GRADIENT COMPARISON: df/dl_MT")
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
        samples = generate_active_fv_samples(
            num_samples=100,
            target_regimes=["CONCENTRIC", "ISOMETRIC", "ECCENTRIC"],  # Avoid extreme eccentric
            velocity_bounds=(-1.5, 1.5)  # Controlled velocity range
        )
        
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
            
            # Compare with l_M gradients (conceptual comparison)
            print("\nKEY INSIGHTS for df/dl_MT vs df/dl_M:")
            print("1. df/dl_MT gradients should generally be smaller in magnitude")
            print("2. df/dl_MT computation is simpler (no pennation angle derivative)")
            print("3. df/dl_MT reflects tendon elasticity effects more directly")
            print("4. Both gradients are important for musculotendon system analysis")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()