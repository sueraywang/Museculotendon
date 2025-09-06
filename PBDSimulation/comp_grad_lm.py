import numpy as np
import matplotlib.pyplot as plt
from Physics import *

# Parameters
act = 1.0
h = 1e-4

# Precompute constants for faster calculations
fMopt = params['fMopt']
beta = params['beta']
alphaMopt = params['alphaMopt']
inv_k_l = 1.0 / params['lMopt']
inv_k_v_h = 1.0 / (params['lMopt'] * params['vMmax'] * h)

def muscle_function(lMtilde, vMtilde):
    afl = curves['AFL'].calc_value(lMtilde)
    pfl = curves['PFL'].calc_value(lMtilde)
    fv = curves['FV'].calc_value(vMtilde)
    penn = np.arcsin(np.sin(alphaMopt)/lMtilde)
    f_m = act * afl * fv + pfl + beta * vMtilde
    
    return fMopt * f_m * np.cos(penn)

def analytical_gradient(lMtilde, vMtilde):
    """
    Compute the analytical gradient of the muscle function
    Returns: [∂F/∂lMtilde, ∂F/∂vMtilde]
    """
    # Get values and derivatives from curves
    res_afl = curves['AFL'].calc_val_deriv(lMtilde)
    afl = res_afl[0]
    afl_deriv = res_afl[1]  # derivative w.r.t lMtilde
    
    res_pfl = curves['PFL'].calc_val_deriv(lMtilde)
    pfl = res_pfl[0]
    pfl_deriv = res_pfl[1] # derivative w.r.t lMtilde
    
    res_fv = curves['FV'].calc_val_deriv(vMtilde)
    fv = res_fv[0]
    fv_deriv = res_fv[1]  # derivative w.r.t vMtilde
    
    # Compute pennation angle and its derivatives
    penn = np.arcsin(np.sin(alphaMopt)/lMtilde)
    cos_penn = np.cos(penn)
    
    # Compute f_m (muscle force without pennation)
    f_m = act * afl * fv + pfl + beta * vMtilde
    
    # Partial derivatives of f_m
    df_m_dlMtilde = act * afl_deriv * fv + pfl_deriv
    df_m_dvMtilde = act * afl * fv_deriv + beta
    
    # Partial derivative of pennation angle w.r.t lMtilde
    dpenn_dlMtilde = -np.sin(alphaMopt) / (lMtilde**2 * np.sqrt(1 - (np.sin(alphaMopt)/lMtilde)**2))
    
    # Compute total partial derivatives using product rule
    dF_dlMtilde = fMopt * (df_m_dlMtilde * cos_penn - f_m * np.sin(penn) * dpenn_dlMtilde)
    
    # dF/dvMtilde = df_m/dvMtilde * cos(penn) (penn doesn't depend on vMtilde)
    dF_dvMtilde = fMopt * df_m_dvMtilde * cos_penn
    
    return np.array([dF_dlMtilde, dF_dvMtilde])

def finite_difference_gradient(lMtilde, vMtilde, h=1e-5, method='central'):
    """
    Compute gradient using finite differences
    
    Parameters:
    - lMtilde, vMtilde: point at which to evaluate gradient
    - h: step size for finite differences
    - method: 'forward', 'backward', or 'central'
    
    Returns: [∂F/∂lMtilde, ∂F/∂vMtilde]
    """
    if method == 'forward':
        # Forward difference: (f(x+h) - f(x)) / h
        dF_dlMtilde = (muscle_function(lMtilde + h, vMtilde) - 
                       muscle_function(lMtilde, vMtilde)) / h
        dF_dvMtilde = (muscle_function(lMtilde, vMtilde + h) - 
                       muscle_function(lMtilde, vMtilde)) / h
        
    elif method == 'backward':
        # Backward difference: (f(x) - f(x-h)) / h
        dF_dlMtilde = (muscle_function(lMtilde, vMtilde) - 
                       muscle_function(lMtilde - h, vMtilde)) / h
        dF_dvMtilde = (muscle_function(lMtilde, vMtilde) - 
                       muscle_function(lMtilde, vMtilde - h)) / h
        
    elif method == 'central':
        # Central difference: (f(x+h) - f(x-h)) / (2h)
        dF_dlMtilde = (muscle_function(lMtilde + h, vMtilde) - 
                       muscle_function(lMtilde - h, vMtilde)) / (2 * h)
        dF_dvMtilde = (muscle_function(lMtilde, vMtilde + h) - 
                       muscle_function(lMtilde, vMtilde - h)) / (2 * h)
    
    return np.array([dF_dlMtilde, dF_dvMtilde])

def nn_gradient(l_m_tilde, v_m_tilde, model):
    """
    Calculate the full gradient of muscle function using neural network,
    based on the relationship: f_muscle = -f_M^opt * C * ∇_{l_tilde} C
    
    Returns both partial derivatives: [∂f/∂l_tilde, ∂f/∂v_tilde]
    """
    # Enable gradients for this function
    with torch.enable_grad():
        # Create input tensor with requires_grad=True
        inputs = torch.tensor([[l_m_tilde, act, v_m_tilde, alphaMopt]], 
                              requires_grad=True, dtype=torch.float32)
        
        # Forward pass through the neural network to get C
        C = model(inputs)
        
        # Ensure C requires gradients
        if not C.requires_grad:
            C = C.detach().requires_grad_(True)
        
        # Calculate first derivatives of C w.r.t. inputs
        grad_outputs = torch.ones_like(C, requires_grad=False)
        first_grads = torch.autograd.grad(C, inputs, 
                                         grad_outputs=grad_outputs, 
                                         create_graph=True, 
                                         retain_graph=True)
        
        if not first_grads or first_grads[0] is None:
            return np.array([0.0, 0.0])
            
        first_grads = first_grads[0]
        
        # Extract first derivatives
        dC_dl = first_grads[0, 0]  # ∂C/∂l_m_tilde
        dC_dv = first_grads[0, 2]  # ∂C/∂v_m_tilde (skipping act which is at index 1)
        
        # Calculate second derivatives for l_tilde component
        # For ∂f/∂l_tilde: we need ∂²C/∂l²
        d2C_dl = torch.autograd.grad(dC_dl, inputs, 
                                    grad_outputs=torch.ones_like(dC_dl), 
                                    create_graph=True, 
                                    retain_graph=True)
        
        d2C_dldl = d2C_dl[0][0, 0].item()  # ∂²C/∂l²
        
        # Calculate second derivatives for v_tilde component  
        # For ∂f/∂v_tilde: we need ∂²C/∂l∂v
        d2C_dv = torch.autograd.grad(dC_dv, inputs, 
                                    grad_outputs=torch.ones_like(dC_dv), 
                                    create_graph=False, 
                                    retain_graph=False)
        
        d2C_dldv = d2C_dv[0][0, 0].item()  # ∂²C/∂l∂v
        
        # Calculate the muscle function derivatives using the product rule
        # f_muscle = -f_M^opt * C * (∂C/∂l)
        # 
        # ∂f_muscle/∂l = -f_M^opt * [∂C/∂l * ∂C/∂l + C * ∂²C/∂l²]
        # ∂f_muscle/∂v = -f_M^opt * [∂C/∂v * ∂C/∂l + C * ∂²C/∂l∂v]
        
        C_val = C.item()
        dC_dl_val = dC_dl.item()
        dC_dv_val = dC_dv.item()
        
        df_dl = -fMopt * (dC_dl_val * dC_dl_val + C_val * d2C_dldl)
        df_dv = -fMopt * (dC_dv_val * dC_dl_val + C_val * d2C_dldv)
        
        return np.array([df_dl, df_dv])

def compare_three_methods(lMtilde_vals, vMtilde_vals, model, epsilon=1e-5, plot=True):
    """
    Compare analytical, finite difference (forward), and neural network gradients
    
    Parameters:
    - lMtilde_vals, vMtilde_vals: arrays of points to test
    - model: trained neural network model
    - epsilon: step size for finite differences (default: 1e-5)
    - plot: whether to create comparison plots
    """
    results = []
    
    for lM in lMtilde_vals:
        for vM in vMtilde_vals:
            try:
                # Compute gradients using all three methods
                analytical_grad = analytical_gradient(lM, vM)
                fd_grad = finite_difference_gradient(lM, vM, epsilon, 'central')
                nn_grad = nn_gradient(lM, vM, model)
                
                # Calculate errors relative to analytical gradient
                fd_error = np.linalg.norm(fd_grad - analytical_grad)
                nn_error = np.linalg.norm(nn_grad - analytical_grad)
                
                # Relative error between finite difference and neural network
                relative_nn_error = np.linalg.norm(nn_grad - analytical_grad)/ np.linalg.norm(nn_grad)
                
                point_result = {
                    'lMtilde': lM,
                    'vMtilde': vM,
                    'analytical': analytical_grad,
                    'finite_diff': fd_grad,
                    'neural_network': nn_grad,
                    'fd_error': fd_error,
                    'nn_error': nn_error,
                    'relative_nn_error': relative_nn_error
                }
                
                results.append(point_result)
                
            except Exception as e:
                print(f"Error at point (lM={lM:.2f}, vM={vM:.2f}): {e}")
                continue
    
    if plot and results:
        plot_three_methods_comparison(results, epsilon)
    
    return results

def plot_three_methods_comparison(results, epsilon):
    """
    Create plots comparing all three gradient methods
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract data for plotting
    analytical_grad1 = [res['analytical'][0] for res in results]
    analytical_grad2 = [res['analytical'][1] for res in results]
    fd_grad1 = [res['finite_diff'][0] for res in results]
    fd_grad2 = [res['finite_diff'][1] for res in results]
    nn_grad1 = [res['neural_network'][0] for res in results]
    nn_grad2 = [res['neural_network'][1] for res in results]
    
    fd_errors = [res['fd_error'] for res in results]
    nn_errors = [res['nn_error'] for res in results]
    
    # Plot 1: Finite Difference vs Analytical (Component 1)
    scatter1 = axes[0,0].scatter(analytical_grad1, fd_grad1, alpha=0.7, color='blue', 
                                s=50, edgecolors='navy', linewidth=0.5, label='Test Points')
    perfect_line1 = axes[0,0].plot([min(analytical_grad1), max(analytical_grad1)], 
                                  [min(analytical_grad1), max(analytical_grad1)], 
                                  'r--', linewidth=2, label='Perfect Agreement')
    axes[0,0].set_xlabel('Analytical ∂F/∂lMtilde', fontsize=11)
    axes[0,0].set_ylabel('Finite Diff ∂F/∂lMtilde', fontsize=11)
    axes[0,0].set_title(f'Finite Difference vs Analytical\n∂F/∂lMtilde (ε = {epsilon})', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=9)
    
    # Plot 2: Neural Network vs Analytical (Component 1)
    scatter2 = axes[0,1].scatter(analytical_grad1, nn_grad1, alpha=0.7, color='green', 
                                s=50, edgecolors='darkgreen', linewidth=0.5, label='Test Points')
    perfect_line2 = axes[0,1].plot([min(analytical_grad1), max(analytical_grad1)], 
                                  [min(analytical_grad1), max(analytical_grad1)], 
                                  'r--', linewidth=2, label='Perfect Agreement')
    axes[0,1].set_xlabel('Analytical ∂F/∂lMtilde', fontsize=11)
    axes[0,1].set_ylabel('Neural Network ∂F/∂lMtilde', fontsize=11)
    axes[0,1].set_title('Neural Network vs Analytical\n∂F/∂lMtilde', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(fontsize=9)
    
    # Plot 3: Error Comparison
    methods = ['Finite Diff', 'Neural Network']
    errors = [fd_errors, nn_errors]
    box_plot = axes[0,2].boxplot(errors, labels=methods, patch_artist=True)
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0,2].set_ylabel('Gradient Error (L2 norm)', fontsize=11)
    axes[0,2].set_title('Error Comparison', fontsize=12)
    axes[0,2].set_yscale('log')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Finite Difference vs Analytical (Component 2)
    scatter3 = axes[1,0].scatter(analytical_grad2, fd_grad2, alpha=0.7, color='blue', 
                                s=50, edgecolors='navy', linewidth=0.5, label='Test Points')
    perfect_line3 = axes[1,0].plot([min(analytical_grad2), max(analytical_grad2)], 
                                  [min(analytical_grad2), max(analytical_grad2)], 
                                  'r--', linewidth=2, label='Perfect Agreement')
    axes[1,0].set_xlabel('Analytical ∂F/∂vMtilde', fontsize=11)
    axes[1,0].set_ylabel('Finite Diff ∂F/∂vMtilde', fontsize=11)
    axes[1,0].set_title(f'Finite Difference vs Analytical\n∂F/∂vMtilde (ε = {epsilon})', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend(fontsize=9)
    
    # Plot 5: Neural Network vs Analytical (Component 2)
    scatter4 = axes[1,1].scatter(analytical_grad2, nn_grad2, alpha=0.7, color='green', 
                                s=50, edgecolors='darkgreen', linewidth=0.5, label='Test Points')
    perfect_line4 = axes[1,1].plot([min(analytical_grad2), max(analytical_grad2)], 
                                  [min(analytical_grad2), max(analytical_grad2)], 
                                  'r--', linewidth=2, label='Perfect Agreement')
    axes[1,1].set_xlabel('Analytical ∂F/∂vMtilde', fontsize=11)
    axes[1,1].set_ylabel('Neural Network ∂F/∂vMtilde', fontsize=11)
    axes[1,1].set_title('Neural Network vs Analytical\n∂F/∂vMtilde', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(fontsize=9)
    
    # Plot 6: REPLACE the existing scatter plot with this:
    relative_nn_error = [res['relative_nn_error'] for res in results]  
    
    # Create histogram instead of scatter plot
    axes[1,2].hist(relative_nn_error, bins=20, alpha=0.7, color='orange', 
                   edgecolor='darkorange', linewidth=1)
    axes[1,2].set_xlabel('norm(nn - analytical)/norm(nn)', fontsize=11)
    axes[1,2].set_ylabel('Frequency', fontsize=11)
    axes[1,2].set_title('Distribution of Relative NN Error', fontsize=12)
    axes[1,2].grid(True, alpha=0.3)
    
    # Add statistics text box
    mean_fd_nn = np.mean(relative_nn_error)
    axes[1,2].axvline(mean_fd_nn, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_fd_nn:.2e}')
    axes[1,2].legend(fontsize=9)
    
    # Add overall title
    fig.suptitle('Gradient Computation Comparison', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the main title
    plt.savefig('GradientComparison_lm_withPenn.png')
    plt.show()

def print_detailed_comparison(results):
    """
    Print detailed comparison of all three methods
    """    
    if results:
        fd_errors = [res['fd_error'] for res in results]
        nn_errors = [res['nn_error'] for res in results]
        relative_nn_error = [res['relative_nn_error'] for res in results]
        
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"Finite Difference (ε = 1e-5):")
        print(f"  Mean error:    {np.mean(fd_errors):.2e}")
        print(f"  Median error:  {np.median(fd_errors):.2e}")
        print(f"  Max error:     {np.max(fd_errors):.2e}")
        print(f"  Min error:     {np.min(fd_errors):.2e}")
        
        print(f"\nNeural Network:")
        print(f"  Mean error:    {np.mean(nn_errors):.2e}")
        print(f"  Median error:  {np.median(nn_errors):.2e}")
        print(f"  Max error:     {np.max(nn_errors):.2e}")
        print(f"  Min error:     {np.min(nn_errors):.2e}")
        
        print(f"\nnorm(nn - analytical)/norm(nn):")
        print(f"  Mean error:    {np.mean(relative_nn_error):.2e}")
        print(f"  Median error:  {np.median(relative_nn_error):.2e}")
        print(f"  Max error:     {np.max(relative_nn_error):.2e}")
        print(f"  Min error:     {np.min(relative_nn_error):.2e}")

# Example usage:
if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = MLP(input_size=model_params['input_size'], hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
    
    # Test points
    lMtilde_test = np.linspace(0.5, 1.5, 10)
    vMtilde_test = np.linspace(-1.0, 1.0, 10)
    
    # Use try-except for better error handling
    try:
        # Load the checkpoint
        checkpoint = torch.load(os.path.join('TrainedModels/Muscles', model_params['model_name']), map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set model to evaluation mode
        print("Comparing analytical, finite difference (forward), and neural network gradients...")
        results = compare_three_methods(lMtilde_test, vMtilde_test, model, epsilon=1e-5)
        print_detailed_comparison(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()