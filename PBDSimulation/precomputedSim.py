# SimulatorNN_Precomputed.py
from Physics import *

model = MLP(input_size=7, hidden_size=128, 
            output_size=1, num_layers=4, 
            activation='gelu')
checkpoint = torch.load(os.path.join('TrainedModels/Muscles', 'precomputed_model_with_lMtilde.pth'   ))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def compute_precomputed_features_with_gradients(lMtilde, activation, vMtilde, alphaMopt):
    """
    Compute pre-computed features with gradient computation for physics loss
    Returns features and the gradient of C w.r.t. lMtilde using chain rule
    """
    # Ensure lMtilde is within valid bounds
    lMtilde = max(lMtilde, 1e-5)
    
    # Pre-compute curve values and their derivatives
    res_afl = curves['AFL'].calc_val_deriv(lMtilde)
    afl = res_afl[0]
    dafl_dlMtilde = res_afl[1]
    
    res_pfl = curves['PFL'].calc_val_deriv(lMtilde)
    pfl = res_pfl[0]
    dpfl_dlMtilde = res_pfl[1]
    
    res_fv = curves['FV'].calc_val_deriv(vMtilde)
    fv = res_fv[0]
    dfv_dvMtilde = res_fv[1]
    
    # Pre-compute pennation angle cosine and its derivative
    sin_alpha = np.sin(alphaMopt)
    penn = np.arcsin(sin_alpha / lMtilde)
    cos_penn = np.cos(penn)
    sin_penn = np.sin(penn)
    dcos_penn_dlMtilde = sin_penn * sin_alpha / (lMtilde**2 * np.sqrt(1 - (sin_alpha/lMtilde)**2))
    
    # Create input tensor [act, afl, fv, pfl, cos_penn, vMtilde]
    act_tensor = torch.tensor(activation, dtype=torch.float32, requires_grad=False)
    afl_tensor = torch.tensor(afl, dtype=torch.float32, requires_grad=True)
    fv_tensor = torch.tensor(fv, dtype=torch.float32, requires_grad=True)
    pfl_tensor = torch.tensor(pfl, dtype=torch.float32, requires_grad=True)
    cos_penn_tensor = torch.tensor(cos_penn, dtype=torch.float32, requires_grad=True)
    vMtilde_tensor = torch.tensor(vMtilde, dtype=torch.float32, requires_grad=True)
    lMtilde_tensor = torch.tensor(lMtilde, dtype=torch.float32, requires_grad=True)
    
    features = torch.stack([act_tensor, afl_tensor, fv_tensor, pfl_tensor, cos_penn_tensor, vMtilde_tensor, lMtilde_tensor]).reshape(1, -1)
    
    # Forward pass
    C = model(features)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=C,
        inputs=[afl_tensor, fv_tensor, pfl_tensor, cos_penn_tensor, vMtilde_tensor, lMtilde_tensor],
        grad_outputs=torch.ones_like(C),
        create_graph=False,
        retain_graph=False
    )
    
    dC_dafl = gradients[0].item()
    dC_dfv = gradients[1].item()
    dC_dpfl = gradients[2].item()
    dC_dcos_penn = gradients[3].item()
    dC_dvMtilde = gradients[4].item()
    
    # Compute dC_dlMtilde using chain rule
    # dC_dlMtilde = dC_dafl * dafl_dlMtilde + dC_dpfl * dpfl_dlMtilde + dC_dcos_penn * dcos_penn_dlMtilde
    dC_dlMtilde = (dC_dafl * dafl_dlMtilde + 
                   dC_dpfl * dpfl_dlMtilde + 
                   dC_dcos_penn * dcos_penn_dlMtilde)
    
    return C.item(), dC_dlMtilde, penn

class Simulator:
    def __init__(self):
        self.dt = DT
        self.num_substeps = SUB_STEPS
        self.sub_dt = self.dt / self.num_substeps
        self.gravity = GRAVITY
        self.penn = 0.0

        self.particles = [
            Particle(XPBD_FIX_POS.copy(), fixed=True, xpbd=True),   
            Particle(XPBD_FREE_POS.copy(), xpbd=True),
            Particle(CLASSIC_FIX_POS.copy(), fixed=True),
            Particle(CLASSIC_FREE_POS.copy())
        ]

        self.constraints = [
            Constraint(self.particles[0], self.particles[1], MUSCLE_COMPLIANCE)
        ]

    def step(self, activation):
        for _ in range(self.num_substeps):
            penn1 = self.xpbd_substep(activation)
            penn2 = self.classic_substep(activation)
            #if penn1 - penn2 < 1e-3:
            self.penn = penn1

    def xpbd_substep(self, activation):
        fixed_particle = self.particles[0]
        moving_particle = self.particles[1]
        moving_particle.prev_position = moving_particle.position.copy()
        moving_particle.velocity += self.gravity * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt

        for constraint in self.constraints:
            dx = fixed_particle.position - moving_particle.position
            dx_prev = fixed_particle.prev_position - moving_particle.prev_position
            relative_motion = dx - dx_prev
            relative_vel = fixed_particle.velocity - moving_particle.velocity
            n = dx / np.linalg.norm(dx)

            w2 = moving_particle.weight
            alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            beta = -params['beta'] * self.sub_dt**2 / params['lMopt'] / params['vMmax'] * params['fMopt']# * math.cos(penn)
            
            # Compute normalized inputs
            lMtilde = np.linalg.norm(dx) / params['lMopt']
            vMtilde = np.dot(relative_vel, normalized(dx)) / (params['lMopt'] * params['vMmax'])
            
            if lMtilde < 0 or lMtilde > 2:
                print(f"lMtilde exceeds the bound. It's {lMtilde}") 
            
            res_afl = curves['AFL'].calc_val_deriv(lMtilde)
            afl = res_afl[0]
            dafl_dlMtilde = res_afl[1]
            
            res_pfl = curves['PFL'].calc_val_deriv(lMtilde)
            pfl = res_pfl[0]
            dpfl_dlMtilde = res_pfl[1]
            
            res_fv = curves['FV'].calc_val_deriv(vMtilde)
            fv = res_fv[0]
            dfv_dvMtilde = res_fv[1]
            
            # Pre-compute pennation angle cosine and its derivative
            sin_alpha = np.sin(params['alphaMopt'])
            penn = np.arcsin(sin_alpha / lMtilde)
            cos_penn = np.cos(penn)
            sin_penn = np.sin(penn)
            dcos_penn_dlMtilde = sin_penn * sin_alpha / (lMtilde**2 * np.sqrt(1 - (sin_alpha/lMtilde)**2))
            
            # Create tensors for the 7 inputs
            act_tensor = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
            afl_tensor = torch.tensor(afl, dtype=torch.float32).reshape(-1, 1)
            fv_tensor = torch.tensor(fv, dtype=torch.float32).reshape(-1, 1)
            pfl_tensor = torch.tensor(pfl, dtype=torch.float32).reshape(-1, 1)
            cos_penn_tensor = torch.tensor(cos_penn, dtype=torch.float32).reshape(-1, 1)
            vMtilde_tensor = torch.tensor(vMtilde, dtype=torch.float32).reshape(-1, 1)
            lMtilde_tensor = torch.tensor(lMtilde, dtype=torch.float32).reshape(-1, 1)
            
            # Derivative tensors for chain rule
            dafl_dlMtilde_tensor = torch.tensor(dafl_dlMtilde, dtype=torch.float32).reshape(-1, 1)
            dpfl_dlMtilde_tensor = torch.tensor(dpfl_dlMtilde, dtype=torch.float32).reshape(-1, 1)
            
            # Method 1: Compute gradients the SAME way as training
            # Enable gradients for variables we need derivatives of
            afl_grad = afl_tensor.detach().requires_grad_(True)
            pfl_grad = pfl_tensor.detach().requires_grad_(True)
            lMtilde_grad = lMtilde_tensor.detach().requires_grad_(True)
            
            # Create network input with gradient-enabled variables
            network_inputs = torch.cat([
                act_tensor, afl_grad, fv_tensor, pfl_grad, 
                cos_penn_tensor, vMtilde_tensor, lMtilde_grad
            ], dim=1)
            
            # Forward pass
            C = model(network_inputs)
            
            # Compute partial derivatives
            dC_dafl = torch.autograd.grad(C, afl_grad, retain_graph=True)[0]
            dC_dpfl = torch.autograd.grad(C, pfl_grad, retain_graph=True)[0]
            dC_dlMtilde_direct = torch.autograd.grad(C, lMtilde_grad, retain_graph=True)[0]
            
            # Total derivative using chain rule (SAME AS TRAINING)
            grad = (dC_dlMtilde_direct + dC_dafl * dafl_dlMtilde_tensor + dC_dpfl * dpfl_dlMtilde_tensor).item()
            
            denominator = (w2 * np.dot(grad, grad)) + alpha
            
            if denominator == 0:
                continue
            
            # Solve constraint: delta_lambda = -C / denominator
            delta_lambda = -C.item() / denominator# * math.cos(penn)
            moving_particle.position += w2 * delta_lambda * grad * n
            
            # Update velocity
            moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt
            
        return penn

    def classic_substep(self, activation):
        fixed_particle = self.particles[2]
        moving_particle = self.particles[3]
        dx = fixed_particle.position - moving_particle.position
        penn = np.arcsin(params['h'] / np.linalg.norm(dx))
        relative_vel = fixed_particle.velocity - moving_particle.velocity
        lMtilde = np.linalg.norm(dx) / params['lMopt']
        vMtilde = np.dot(relative_vel, normalized(dx)) / (params['lMopt'] * params['vMmax'])
        
        f_muscle = muscle_force_full(lMtilde, activation, vMtilde, penn) * params['fMopt'] * normalized(dx)

        # Update velocity and position
        moving_particle.velocity += ((f_muscle + moving_particle.mass * self.gravity) / moving_particle.mass) * self.sub_dt
        moving_particle.position += moving_particle.velocity * self.sub_dt
        
        return penn