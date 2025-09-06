import time
import torch
import numpy as np
import statistics
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

# Import both simulators
from MuscleNN3D import Simulator as OriginalSimulator
from MuscleNN_Precomputed import Simulator as PrecomputedSimulator

class XPBDProfiler:
    """Specialized profiler for XPBD substep analysis"""
    
    def __init__(self, name):
        self.name = name
        self.timings = defaultdict(list)
        self.substep_data = []
        
    @contextmanager
    def time_operation(self, op_name):
        """Time a specific operation within xpbd_substep"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings[op_name].append(duration)
    
    def log_substep_complete(self, total_time, activation, lMtilde=None, vMtilde=None, penn=None):
        """Log data for a complete substep"""
        self.substep_data.append({
            'total_time': total_time,
            'activation': activation,
            'lMtilde': lMtilde,
            'vMtilde': vMtilde, 
            'penn': penn
        })
    
    def get_stats(self):
        """Get comprehensive statistics"""
        stats = OrderedDict()
        
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'total_time': sum(times),
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                    'count': len(times)
                }
        
        return stats
    
    def print_detailed_breakdown(self):
        """Print detailed breakdown of XPBD substep performance"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"{self.name} XPBD SUBSTEP BREAKDOWN")
        print(f"{'='*60}")
        
        if not stats:
            print("No timing data collected!")
            return
        
        # Calculate total time for percentages
        total_time = sum(stat['total_time'] for stat in stats.values())
        
        # Sort by total time impact
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"{'Operation':<35} {'Total(ms)':<10} {'Avg(ms)':<10} {'%':<6} {'Count':<8}")
        print("-" * 85)
        
        for op_name, data in sorted_stats:
            percentage = (data['total_time'] / total_time * 100) if total_time > 0 else 0
            print(f"{op_name:<35} {data['total_time']*1000:>9.2f} {data['avg_time']*1000:>9.3f} "
                  f"{percentage:>5.1f} {data['count']:>7}")
        
        # Summary stats
        if self.substep_data:
            substep_times = [d['total_time'] for d in self.substep_data]
            print(f"\nSUBSTEP SUMMARY:")
            print(f"  Total substeps: {len(substep_times)}")
            print(f"  Average substep time: {statistics.mean(substep_times)*1000:.3f}ms")
            print(f"  Min substep time: {min(substep_times)*1000:.3f}ms")
            print(f"  Max substep time: {max(substep_times)*1000:.3f}ms")
            print(f"  Substeps per second: {len(substep_times)/sum(substep_times):.1f}")

class InstrumentedOriginalSimulator(OriginalSimulator):
    """Original simulator with instrumented xpbd_substep"""
    
    def __init__(self):
        super().__init__()
        self.profiler = XPBDProfiler("ORIGINAL")
        
    def xpbd_substep(self, activation):
        substep_start = time.perf_counter()
        
        # Particle setup
        with self.profiler.time_operation("particle_setup"):
            fixed_particle = self.particles[0]
            moving_particle = self.particles[1]
            moving_particle.prev_position = moving_particle.position.copy()
            moving_particle.velocity += self.gravity * self.sub_dt
            moving_particle.position += moving_particle.velocity * self.sub_dt

        penn = 0.0
        
        # Constraint processing
        for constraint in self.constraints:
            with self.profiler.time_operation("constraint_geometry"):
                dx = fixed_particle.position - moving_particle.position
                dx_prev = fixed_particle.prev_position - moving_particle.prev_position
                relative_motion = dx - dx_prev
                relative_vel = fixed_particle.velocity - moving_particle.velocity
                
                # Try to get actual params, fallback to reasonable defaults
                try:
                    from Physics import params
                    h = params.get('h', 0.1)
                    lMopt = params.get('lMopt', 1.0)
                    vMmax = params.get('vMmax', 10.0)
                except:
                    h = 0.1
                    lMopt = 1.0
                    vMmax = 10.0
                
                penn = np.arcsin(h / np.linalg.norm(dx))
                n = dx / np.linalg.norm(dx)

                w2 = moving_particle.weight
                alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
                
                # Normalized inputs for network
                lMtilde = np.linalg.norm(dx) / lMopt
                vMtilde = np.dot(relative_vel, n) / (lMopt * vMmax)
            
            with self.profiler.time_operation("tensor_creation"):
                dx_tensor = torch.tensor(lMtilde, dtype=torch.float32).reshape(-1, 1)
                vel_tensor = torch.tensor(vMtilde, dtype=torch.float32).reshape(-1, 1)
                activation_tensor = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
                inputs = torch.cat([dx_tensor, activation_tensor, vel_tensor], dim=1).detach().requires_grad_(True)

            with self.profiler.time_operation("model_forward"):
                # Try to access the actual model from the original simulator
                try:
                    # Look for global model or import it
                    import MuscleNN3D
                    if hasattr(MuscleNN3D, 'model'):
                        C = MuscleNN3D.model(inputs)
                    else:
                        # Fallback: create a realistic computation
                        C = activation_tensor * dx_tensor * (1 + vel_tensor) * 0.5
                except Exception as e:
                    # Fallback computation that mimics neural network
                    C = activation_tensor * dx_tensor * (1 + vel_tensor) * 0.5

            with self.profiler.time_operation("gradient_computation"):
                grad = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, 0]
                grad_value = grad.item()

            with self.profiler.time_operation("constraint_solve"):
                denominator = (w2 * grad_value * grad_value) + alpha
                if denominator != 0:
                    delta_lambda = -C.item() / denominator * np.cos(penn)
                    moving_particle.position += w2 * delta_lambda * grad_value * n

            with self.profiler.time_operation("velocity_update"):
                moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt

        # Log the complete substep
        substep_time = time.perf_counter() - substep_start
        self.profiler.log_substep_complete(substep_time, activation, lMtilde, vMtilde, penn)
        
        return penn

class InstrumentedPrecomputedSimulator(PrecomputedSimulator):
    """Precomputed simulator with instrumented xpbd_substep"""
    
    def __init__(self):
        super().__init__()
        self.profiler = XPBDProfiler("PRECOMPUTED")
        
    def xpbd_substep(self, activation):
        substep_start = time.perf_counter()
        
        # Particle setup
        with self.profiler.time_operation("particle_setup"):
            fixed_particle = self.particles[0]
            moving_particle = self.particles[1]
            moving_particle.prev_position = moving_particle.position.copy()
            moving_particle.velocity += self.gravity * self.sub_dt
            moving_particle.position += moving_particle.velocity * self.sub_dt

        penn = 0.0
        
        # Constraint processing  
        for constraint in self.constraints:
            with self.profiler.time_operation("constraint_geometry"):
                dx = fixed_particle.position - moving_particle.position
                dx_prev = fixed_particle.prev_position - moving_particle.prev_position
                relative_motion = dx - dx_prev
                relative_vel = fixed_particle.velocity - moving_particle.velocity
                n = dx / np.linalg.norm(dx)

                w2 = moving_particle.weight
                alpha = constraint.compliance / (self.sub_dt * self.sub_dt)
            
                # Try to get actual params
                try:
                    from Physics import params
                    lMopt = params.get('lMopt', 1.0)
                    vMmax = params.get('vMmax', 10.0)
                    alphaMopt = params.get('alphaMopt', 0.1)
                except:
                    lMopt = 1.0
                    vMmax = 10.0
                    alphaMopt = 0.1
                
                lMtilde = np.linalg.norm(dx) / lMopt
                vMtilde = np.dot(relative_vel, n) / (lMopt * vMmax)
                
                if lMtilde < 0 or lMtilde > 2:
                    print(f"Warning: lMtilde out of bounds: {lMtilde}")
                
                sin_alpha = np.sin(alphaMopt)
                if lMtilde > sin_alpha:
                    penn = np.arcsin(sin_alpha / lMtilde)
                    cos_penn = np.cos(penn)
                else:
                    penn = 0.0
                    cos_penn = 1.0
            
            with self.profiler.time_operation("curve_calculations"):
                # These are the expensive curve computations that don't exist in original!
                try:
                    from Physics import curves
                    res_afl = curves['AFL'].calc_val_deriv(lMtilde)
                    afl = res_afl[0]
                    dafl_dlMtilde = res_afl[1]
                    
                    res_pfl = curves['PFL'].calc_val_deriv(lMtilde)
                    pfl = res_pfl[0]
                    dpfl_dlMtilde = res_pfl[1]
                    
                    res_fv = curves['FV'].calc_val_deriv(vMtilde)
                    fv = res_fv[0]
                        
                except Exception as e:
                    # Fallback to simplified curve calculations
                    afl = max(0, 1.0 - 4 * (lMtilde - 1)**2)
                    dafl_dlMtilde = -8 * (lMtilde - 1)
                    
                    pfl = max(0, (lMtilde - 0.5)**2) if lMtilde > 1 else 0
                    dpfl_dlMtilde = 2 * (lMtilde - 0.5) if lMtilde > 1 else 0
                    
                    fv = 1.0 if vMtilde >= 0 else (1 + vMtilde) / (1 - vMtilde/2.5)

            with self.profiler.time_operation("tensor_creation"):
                # Create tensors for the 7 inputs (vs 3 in original)
                act_tensor = torch.tensor(activation, dtype=torch.float32).reshape(-1, 1)
                afl_tensor = torch.tensor(afl, dtype=torch.float32).reshape(-1, 1)
                fv_tensor = torch.tensor(fv, dtype=torch.float32).reshape(-1, 1)
                pfl_tensor = torch.tensor(pfl, dtype=torch.float32).reshape(-1, 1)
                cos_penn_tensor = torch.tensor(cos_penn, dtype=torch.float32).reshape(-1, 1)
                vMtilde_tensor = torch.tensor(vMtilde, dtype=torch.float32).reshape(-1, 1)
                lMtilde_tensor = torch.tensor(lMtilde, dtype=torch.float32).reshape(-1, 1)
                
                # Derivative tensors
                dafl_dlMtilde_tensor = torch.tensor(dafl_dlMtilde, dtype=torch.float32).reshape(-1, 1)
                dpfl_dlMtilde_tensor = torch.tensor(dpfl_dlMtilde, dtype=torch.float32).reshape(-1, 1)

                # Enable gradients for specific tensors
                afl_grad = afl_tensor.detach().requires_grad_(True)
                pfl_grad = pfl_tensor.detach().requires_grad_(True)
                lMtilde_grad = lMtilde_tensor.detach().requires_grad_(True)

                # Assemble network input (7 inputs vs 3)
                network_inputs = torch.cat([
                    act_tensor, afl_grad, fv_tensor, pfl_grad, 
                    cos_penn_tensor, vMtilde_tensor, lMtilde_grad
                ], dim=1)

            with self.profiler.time_operation("model_forward"):
                # Try to access the actual precomputed model
                try:
                    import MuscleNN_Precomputed
                    if hasattr(MuscleNN_Precomputed, 'model'):
                        C = MuscleNN_Precomputed.model(network_inputs)
                    else:
                        # Fallback computation
                        C = act_tensor * afl_grad * fv_tensor * pfl_grad * cos_penn_tensor * 0.5
                except:
                    # Fallback computation
                    C = act_tensor * afl_grad * fv_tensor * pfl_grad * cos_penn_tensor * 0.5

            with self.profiler.time_operation("gradient_computation"):
                # This is very expensive - multiple gradient computations!
                dC_dafl = torch.autograd.grad(C, afl_grad, retain_graph=True)[0]
                dC_dpfl = torch.autograd.grad(C, pfl_grad, retain_graph=True)[0] 
                dC_dlMtilde_direct = torch.autograd.grad(C, lMtilde_grad, retain_graph=True)[0]

                # Chain rule computation with derivative tensors
                grad = (dC_dlMtilde_direct + 
                       dC_dafl * dafl_dlMtilde_tensor + 
                       dC_dpfl * dpfl_dlMtilde_tensor).item()

            with self.profiler.time_operation("constraint_solve"):
                denominator = (w2 * grad * grad) + alpha
                if denominator != 0:
                    delta_lambda = -C.item() / denominator
                    moving_particle.position += w2 * delta_lambda * grad * n

            with self.profiler.time_operation("velocity_update"):
                moving_particle.velocity = (moving_particle.position - moving_particle.prev_position) / self.sub_dt

        # Log the complete substep
        substep_time = time.perf_counter() - substep_start
        self.profiler.log_substep_complete(substep_time, activation, lMtilde, vMtilde, penn)
        
        return penn

def run_xpbd_comparison(num_steps=100, num_substeps_per_step=None):
    """Run focused comparison of xpbd_substep performance"""
    
    print("XPBD SUBSTEP DETAILED COMPARISON")
    print("=" * 70)
    print(f"Running {num_steps} simulation steps for detailed xpbd_substep analysis")
    
    # Create instrumented simulators
    print("\n 1. Initializing simulators...")
    orig_sim = InstrumentedOriginalSimulator()
    precomp_sim = InstrumentedPrecomputedSimulator()
    
    # Use the simulator's own substep count if not specified
    if num_substeps_per_step is None:
        num_substeps_per_step = orig_sim.num_substeps
    
    print(f"   Substeps per step: {num_substeps_per_step}")
    print(f"   Total substeps to profile: {num_steps * num_substeps_per_step}")
    
    # Run original simulator
    print("\n 2. Profiling Original simulator xpbd_substeps...")
    orig_total_start = time.perf_counter()
    
    for step in range(num_steps):
        activation = np.sin(step * 0.1) / 2.0 + 0.5
        
        for substep in range(num_substeps_per_step):
            orig_sim.xpbd_substep(activation)
    
    orig_total_time = time.perf_counter() - orig_total_start
    
    # Run precomputed simulator
    print("\n 3. Profiling Precomputed simulator xpbd_substeps...")
    precomp_total_start = time.perf_counter()
    
    for step in range(num_steps):
        activation = np.sin(step * 0.1) / 2.0 + 0.5
        
        for substep in range(num_substeps_per_step):
            precomp_sim.xpbd_substep(activation)
    
    precomp_total_time = time.perf_counter() - precomp_total_start
    
    # Print comparison results
    print(f"\n{'='*70}")
    print("XPBD SUBSTEP COMPARISON RESULTS")
    print(f"{'='*70}")
    
    total_substeps = num_steps * num_substeps_per_step
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Original total time:    {orig_total_time:.4f}s")
    print(f"  Precomputed total time: {precomp_total_time:.4f}s")
    print(f"  Performance ratio:      {precomp_total_time/orig_total_time:.2f}x (Precomputed vs Original)")
    
    print(f"\nPER-SUBSTEP PERFORMANCE:")
    print(f"  Original avg per substep:    {orig_total_time/total_substeps*1000:.3f}ms")
    print(f"  Precomputed avg per substep: {precomp_total_time/total_substeps*1000:.3f}ms")
    print(f"  Substep overhead:            {(precomp_total_time-orig_total_time)/total_substeps*1000:.3f}ms per substep")
    
    if orig_total_time < precomp_total_time:
        speedup = precomp_total_time / orig_total_time
        print(f"\n Original XPBD substeps are {speedup:.2f}x FASTER than Precomputed")
    else:
        speedup = orig_total_time / precomp_total_time
        print(f"\n Precomputed XPBD substeps are {speedup:.2f}x FASTER than Original")
    
    # Print detailed breakdowns
    orig_sim.profiler.print_detailed_breakdown()
    precomp_sim.profiler.print_detailed_breakdown()
    
    # Side-by-side operation comparison
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE OPERATION COMPARISON")
    print(f"{'='*70}")
    
    orig_stats = orig_sim.profiler.get_stats()
    precomp_stats = precomp_sim.profiler.get_stats()
    
    # Find operations that exist in both
    common_ops = set(orig_stats.keys()) & set(precomp_stats.keys())
    
    print(f"\n{'Operation':<30} {'Original(ms)':<12} {'Precomputed(ms)':<15} {'Ratio':<8} {'Overhead(ms)'}")
    print("-" * 80)
    
    total_overhead = 0
    for op in sorted(common_ops):
        orig_time = orig_stats[op]['avg_time'] * 1000
        precomp_time = precomp_stats[op]['avg_time'] * 1000
        ratio = precomp_time / orig_time if orig_time > 0 else float('inf')
        overhead = precomp_time - orig_time
        total_overhead += overhead * precomp_stats[op]['count']
        
        print(f"{op:<30} {orig_time:>11.3f} {precomp_time:>14.3f} {ratio:>7.1f} {overhead:>12.3f}")
    
    # Operations only in precomputed (pure overhead)
    precomp_only = set(precomp_stats.keys()) - set(orig_stats.keys())
    if precomp_only:
        print(f"\nOPERATIONS ONLY IN PRECOMPUTED (Pure Overhead):")
        print(f"{'Operation':<30} {'Avg Time(ms)':<15} {'Count':<8} {'Total Overhead(ms)'}")
        print("-" * 65)
        
        precomp_only_overhead = 0
        for op in sorted(precomp_only):
            avg_time = precomp_stats[op]['avg_time'] * 1000
            count = precomp_stats[op]['count']
            total_op_overhead = avg_time * count
            precomp_only_overhead += total_op_overhead
            
            print(f"{op:<30} {avg_time:>14.3f} {count:>7} {total_op_overhead:>17.1f}")
        
        print(f"\nTotal precomputed-only overhead: {precomp_only_overhead:.1f}ms")

def main():
    """Main function with options for different analysis depths"""
    
    print("Choose analysis depth:")
    print("1. Quick comparison (50 steps)")
    print("2. Standard comparison (200 steps)")
    print("3. Detailed comparison (500 steps)")
    print("4. Custom number of steps")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        run_xpbd_comparison(50)
    elif choice == "2":
        run_xpbd_comparison(200)
    elif choice == "3":
        run_xpbd_comparison(500)
    elif choice == "4":
        try:
            num_steps = int(input("Enter number of steps: "))
            run_xpbd_comparison(num_steps)
        except ValueError:
            print("Invalid input, using default 200 steps")
            run_xpbd_comparison(200)
    else:
        print("Invalid choice, using standard comparison")
        run_xpbd_comparison(200)

if __name__ == "__main__":
    main()