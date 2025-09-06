import numpy as np
import matplotlib.pyplot as plt
from Physics import *

# Parameters
act = 1.0
h = 1e-4

# Precompute constants for faster calculations
fMopt = params['fMopt']
beta = params['beta']
k_l = params['lMopt']  # kl in the PDF
k_v = params['lMopt'] * params['vMmax']  # kv in the PDF
inv_k_l = 1.0 / k_l
inv_k_v_h = 1.0 / (k_v * h)

def muscle_function_x(x, x_0):
    """Original muscle function using x and x_0"""
    lMtilde = x * inv_k_l
    vMtilde = (x - x_0) * inv_k_v_h
    
    afl = curves['AFL'].calc_value(lMtilde)
    pfl = curves['PFL'].calc_value(lMtilde)
    fv = curves['FV'].calc_value(vMtilde)
    
    return (act * afl * fv + pfl + beta * vMtilde) * fMopt

def analytical_gradient_x_vars(x, x_0):
    """
    Compute the analytical gradient of the muscle function with respect to x
    """
    # Convert to normalized variables
    lMtilde = x / k_l
    vMtilde = (x - x_0) / (k_v * h)
    
    # Get values and derivatives from curves
    res_afl = curves['AFL'].calc_val_deriv(lMtilde)
    afl = res_afl[0]
    afl_deriv = res_afl[1]  # dfL/dlM
    
    pfl_deriv = curves['PFL'].calc_derivative(lMtilde, 1)  # dfP/dlM
    
    res_fv = curves['FV'].calc_val_deriv(vMtilde)
    fv = res_fv[0]
    fv_deriv = res_fv[1]  # dfV/dvM
    
    # Fixed derivatives matching velocity definition: vMtilde = (x - x_0) / (k_v * h)
    # So: ∂vMtilde/∂x = +1/(k_v * h)
    term1 = (1.0 / k_l) * (act * fv * afl_deriv + pfl_deriv)
    term2 = (1.0 / (k_v * h)) * (act * afl * fv_deriv + beta)
    
    dF_dx = fMopt * (term1 + term2)  # Note: both terms are positive now
    
    return dF_dx

def richardson_extrapolation_gradient(x, x_0, base_h=None):
    """
    Richardson extrapolation for finite differences
    Uses two step sizes (h and h/2) to cancel O(h²) errors
    """
    if base_h is None:
        base_h = max(1e-7, abs(x) * 1e-7)
    
    # Compute gradients with step size h and h/2
    h1 = base_h
    h2 = base_h / 2
    
    # h step
    f_plus_h1 = muscle_function_x(x + h1, x_0)
    f_minus_h1 = muscle_function_x(x - h1, x_0)
    grad_h1 = (f_plus_h1 - f_minus_h1) / (2 * h1)
    
    # h/2 step  
    f_plus_h2 = muscle_function_x(x + h2, x_0)
    f_minus_h2 = muscle_function_x(x - h2, x_0)
    grad_h2 = (f_plus_h2 - f_minus_h2) / (2 * h2)
    
    # Richardson extrapolation: (4*f'(h/2) - f'(h)) / 3
    richardson_grad = (4 * grad_h2 - grad_h1) / 3
    
    return richardson_grad

def nn_gradient_x_vars(x, x_0, model):
    """
    Calculate the gradient using neural network
    """
    # Convert to normalized variables
    l_m_tilde = x / k_l
    v_m_tilde = (x - x_0) / (k_v * h)
    
    # Enable gradients for this function
    with torch.enable_grad():
        # Create input tensor with requires_grad=True
        inputs = torch.tensor([[l_m_tilde, act, v_m_tilde]], 
                              requires_grad=True, dtype=torch.float32)
        
        # Forward pass through the neural network to get C
        C = model(inputs)
        
        # Calculate first derivatives of C w.r.t. inputs
        grad_outputs = torch.ones_like(C, requires_grad=False)
        first_grads = torch.autograd.grad(C, inputs, 
                                         grad_outputs=grad_outputs, 
                                         create_graph=True, 
                                         retain_graph=True)
        
        if not first_grads or first_grads[0] is None:
            return 0.0
            
        first_grads = first_grads[0]
        
        # Extract first derivatives
        dC_dl = first_grads[0, 0]  # ∂C/∂l_m_tilde
        dC_dv = first_grads[0, 2]  # ∂C/∂v_m_tilde
        
        # Calculate second derivatives
        d2C_dl = torch.autograd.grad(dC_dl, inputs, 
                                    grad_outputs=torch.ones_like(dC_dl), 
                                    create_graph=True, 
                                    retain_graph=True)
        d2C_dldl = d2C_dl[0][0, 0]  # ∂²C/∂l²
        
        d2C_dv = torch.autograd.grad(dC_dv, inputs, 
                                    grad_outputs=torch.ones_like(dC_dv), 
                                    create_graph=False, 
                                    retain_graph=False)
        d2C_dvdl = d2C_dv[0][0, 0]  # ∂²C/∂v∂l
        
        # Calculate the muscle function derivative using equation (26) from PDF
        C_val = C.item()
        dC_dl_val = dC_dl.item()
        dC_dv_val = dC_dv.item()
        d2C_dldl_val = d2C_dldl.item()
        d2C_dvdl_val = d2C_dvdl.item()
        
        # Apply equation (26) with corrected sign
        term1 = (dC_dl_val ** 2) * (1.0 / k_l)
        term2 = (dC_dv_val * dC_dl_val) * (1.0 / (k_v * h))  
        term3 = C_val * d2C_dldl_val * (1.0 / k_l)
        term4 = C_val * d2C_dvdl_val * (1.0 / (k_v * h))
        
        df_dx = -fMopt * (term1 + term2 + term3 + term4)
        
        return df_dx

def compare_three_methods_x_vars(x_vals, x_0_vals, model, plot=True):
    """
    Compare analytical,  finite difference, and neural network gradients
    """
    results = []
    
    # Ensure arrays are the same length
    assert len(x_vals) == len(x_0_vals), "x_vals and x_0_vals must have the same length"
    
    for i in range(len(x_vals)):
        x = x_vals[i]
        x_0 = x_0_vals[i]
        try:
            # Compute gradients using all three methods
            analytical_grad = analytical_gradient_x_vars(x, x_0)
            fd_grad = richardson_extrapolation_gradient(x, x_0)
            nn_grad = nn_gradient_x_vars(x, x_0, model)
            
            # Calculate errors relative to analytical gradient
            fd_error = abs(fd_grad - analytical_grad)
            nn_error = abs(nn_grad - analytical_grad)
            
            # Calculate relative error for neural network
            relative_nn_error = abs(nn_grad - analytical_grad) / (abs(nn_grad) + 1e-15)
            
            point_result = {
                'x': x,
                'x_0': x_0,
                'analytical': analytical_grad,
                'finite_diff': fd_grad,
                'neural_network': nn_grad,
                'fd_error': fd_error,
                'nn_error': nn_error,
                'relative_nn_error': relative_nn_error
            }
            
            results.append(point_result)
            
        except Exception as e:
            print(f"Error at point (x={x:.2f}, x_0={x_0:.2f}): {e}")
            continue
    
    if plot and results:
        plot_three_methods_comparison_x_vars(results)
    
    return results

def plot_three_methods_comparison_x_vars(results):
    """
    Create plots comparing all three gradient methods
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data for plotting
    analytical_grads = [res['analytical'] for res in results]
    fd_grads = [res['finite_diff'] for res in results]
    nn_grads = [res['neural_network'] for res in results]
    
    fd_errors = [res['fd_error'] for res in results]
    nn_errors = [res['nn_error'] for res in results]
    relative_nn_errors = [res['relative_nn_error'] for res in results]
    
    # Plot 1: Finite Difference vs Analytical
    axes[0,0].scatter(analytical_grads, fd_grads, alpha=0.7, color='blue', 
                     s=50, edgecolors='navy', linewidth=0.5, label='Test Points')
    perfect_line_range = [min(analytical_grads), max(analytical_grads)]
    axes[0,0].plot(perfect_line_range, perfect_line_range, 
                   'r--', linewidth=2, label='Perfect Agreement')
    axes[0,0].set_xlabel('Analytical ∂F/∂x', fontsize=11)
    axes[0,0].set_ylabel('Finite Diff ∂F/∂x', fontsize=11)
    axes[0,0].set_title('Finite Difference vs Analytical', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=9)
    
    # Plot 2: Neural Network vs Analytical
    axes[0,1].scatter(analytical_grads, nn_grads, alpha=0.7, color='green', 
                     s=50, edgecolors='darkgreen', linewidth=0.5, label='Test Points')
    axes[0,1].plot(perfect_line_range, perfect_line_range, 
                   'r--', linewidth=2, label='Perfect Agreement')
    axes[0,1].set_xlabel('Analytical ∂F/∂x', fontsize=11)
    axes[0,1].set_ylabel('Neural Network ∂F/∂x', fontsize=11)
    axes[0,1].set_title('Neural Network vs Analytical', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(fontsize=9)
    
    # Plot 3: Error Comparison (Box Plot)
    methods = ['Finite Diff', 'Neural Network']
    errors = [fd_errors, nn_errors]
    box_plot = axes[1,0].boxplot(errors, tick_labels=methods, patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1,0].set_ylabel('Gradient Error (Absolute)', fontsize=11)
    axes[1,0].set_title('Error Comparison', fontsize=12)
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Neural Network Relative Error Distribution
    axes[1,1].hist(np.array(relative_nn_errors), bins=15, alpha=0.7, 
                   color='orange', edgecolor='darkorange', linewidth=1)
    axes[1,1].set_xlabel('Relative NN Error', fontsize=11)
    axes[1,1].set_ylabel('Count', fontsize=11)
    axes[1,1].set_title('Neural Network Relative Error Distribution', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_rel_error = np.mean(relative_nn_errors)
    median_rel_error = np.median(relative_nn_errors)
    axes[1,1].axvline(mean_rel_error, color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {mean_rel_error:.2e}')
    axes[1,1].axvline(median_rel_error, color='black', linestyle='--', 
                     linewidth=2, label=f'Median: {median_rel_error:.2e}')
    axes[1,1].legend(fontsize=9)
    
    # Add overall title
    fig.suptitle('Gradient Computation Comparison (w.r.t. x)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('GradientComparison_xvar.png')
    plt.show()

def print_detailed_comparison_x_vars(results):
    """
    Print detailed comparison
    """    
    if results:
        fd_errors = [res['fd_error'] for res in results]
        nn_errors = [res['nn_error'] for res in results]
        relative_nn_errors = [res['relative_nn_error'] for res in results]
        
        print("\n" + "="*80)
        print("OVERALL STATISTICS (dF/dx)")
        print("="*80)
        print(f"Finite Difference:")
        print(f"  Mean error:    {np.mean(fd_errors):.2e}")
        print(f"  Median error:  {np.median(fd_errors):.2e}")
        print(f"  Max error:     {np.max(fd_errors):.2e}")
        print(f"  Min error:     {np.min(fd_errors):.2e}")
        
        print(f"\nNeural Network (Absolute Error):")
        print(f"  Mean error:    {np.mean(nn_errors):.2e}")
        print(f"  Median error:  {np.median(nn_errors):.2e}")
        print(f"  Max error:     {np.max(nn_errors):.2e}")
        print(f"  Min error:     {np.min(nn_errors):.2e}")
        
        print(f"\nNeural Network (Relative Error):")
        print(f"  Mean rel error:    {np.mean(relative_nn_errors):.2e}")
        print(f"  Median rel error:  {np.median(relative_nn_errors):.2e}")
        print(f"  Max rel error:     {np.max(relative_nn_errors):.2e}")
        print(f"  Min rel error:     {np.min(relative_nn_errors):.2e}")

# Example usage:
if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = MLP(input_size=model_params['input_size'], hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
    
    # Test points - using physical x values instead of normalized ones
    lMtilde_range = np.linspace(0.5, 1.5, 10)
    vMtilde_range = np.linspace(-1.0, 1.0, 10)
    
    # Generate all combinations of x and x_0
    x_test = []
    x_0_test = []
    
    for lMtilde in lMtilde_range:
        x = lMtilde * k_l  # x = lMtilde * k_l
        for vMtilde in vMtilde_range:
            x_0 = x - vMtilde * k_v * h
            x_test.append(x)
            x_0_test.append(x_0)
    
    x_test = np.array(x_test)
    x_0_test = np.array(x_0_test)
    
    # Use try-except for better error handling
    try:
        # Load the checkpoint
        checkpoint = torch.load(os.path.join('TrainedModels/Muscles', model_params['model_name']), map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set model to evaluation mode
        
        print("Comparing analytical, finite difference, and neural network gradients...")
        print(f"Testing {len(x_test)} points")
        print(f"k_l = {k_l:.3e}, k_v = {k_v:.3e}, h = {h:.3e}")
        
        results = compare_three_methods_x_vars(x_test, x_0_test, model)
        print_detailed_comparison_x_vars(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()