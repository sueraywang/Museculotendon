# benchmark.py
import time
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from Simulator import Simulator
from Physics import *

class NeuralNetworkBenchmark:
    def __init__(self, model_configs: List[Dict]):
        """
        Initialize benchmark with list of model configurations
        """
        self.model_configs = model_configs
        self.simulators = {}
        self.timing_results = []
        
        # Initialize muscle length
        if RENDER_MT:
            self.l_M = find_initial_fiber_equilibrium(INITIAL_LENGTH, 0.5)
        else:
            self.l_M = INITIAL_LENGTH
    
    def load_models(self):
        """Load all neural network models and create simulators"""
        print("Loading neural network models...")
        
        for config in self.model_configs:
            try:
                # Create neural network model
                nn_model = MLP(
                    input_size=config['input_size'], 
                    hidden_size=config['num_width'], 
                    output_size=config['output_size'], 
                    num_layers=config['num_layer'], 
                    activation=config['activation_func']
                )
                
                # Load trained weights
                model_path = os.path.join('TrainedModels/Muscles', config['model_name'])
                checkpoint = torch.load(model_path)
                nn_model.load_state_dict(checkpoint['model_state_dict'])
                nn_model.eval()
                
                # Create simulator
                simulator = Simulator(
                    l_M=self.l_M,
                    penn_model=config['includes_penn'],
                    damping_model=config['includes_damp'],
                    nn_model=nn_model
                )
                
                self.simulators[config['model_name']] = {
                    'simulator': simulator,
                    'config': config,
                    'nn_model': nn_model
                }
                
                print(f"✓ Loaded: {config['model_name']}")
                
            except Exception as e:
                print(f"✗ Failed to load {config['model_name']}: {str(e)}")
                continue
    
    def instrument_simulator(self, simulator):
        """Add timing instrumentation to the simulator"""
        # Store original methods
        original_xpbd_substep = simulator.xpbd_substep
        original_nn_forward = simulator.nn_model.forward
        
        # Initialize timing storage
        simulator._substep_times = []
        simulator._nn_forward_times = []
        
        def timed_xpbd_substep(activation):
            start_time = time.perf_counter()
            result = original_xpbd_substep(activation)
            end_time = time.perf_counter()
            simulator._substep_times.append(end_time - start_time)
            return result
        
        def timed_nn_forward(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_nn_forward(*args, **kwargs)
            end_time = time.perf_counter()
            simulator._nn_forward_times.append(end_time - start_time)
            return result
        
        # Replace with instrumented versions
        simulator.xpbd_substep = timed_xpbd_substep
        simulator.nn_model.forward = timed_nn_forward
        
        return simulator
    
    def benchmark_model(self, model_name: str, num_iterations: int = 1000):
        """Benchmark a single model"""
        if model_name not in self.simulators:
            print(f"Model {model_name} not loaded, skipping...")
            return None
        
        print(f"Benchmarking {model_name}...")
        
        # Get fresh simulator instance
        config = self.simulators[model_name]['config']
        nn_model = self.simulators[model_name]['nn_model']
        
        # Create new simulator for benchmarking
        test_simulator = Simulator(
            l_M=self.l_M,
            penn_model=config['includes_penn'],
            damping_model=config['includes_damp'],
            nn_model=nn_model
        )
        
        # Instrument the simulator
        test_simulator = self.instrument_simulator(test_simulator)
        
        # Warm-up runs
        print("  Warming up...")
        for i in range(100):
            activation = 0.5 + 0.3 * np.sin(i * 0.1)
            test_simulator.step(activation)
        
        # Clear timing data after warm-up
        test_simulator._substep_times = []
        test_simulator._nn_forward_times = []
        
        # Actual benchmark
        print(f"  Running {num_iterations} iterations...")
        total_step_times = []
        
        for i in range(num_iterations):
            activation = 0.5 + 0.3 * np.sin(i * 0.1)
            
            # Clear per-step timing
            test_simulator._substep_times = []
            test_simulator._nn_forward_times = []
            
            # Time the full step
            step_start = time.perf_counter()
            test_simulator.step(activation)
            step_end = time.perf_counter()
            
            total_step_times.append(step_end - step_start)
        
        # Calculate statistics
        return {
            'model_name': model_name,
            'config': config,
            'mean_step_time': np.mean(total_step_times),
            'std_step_time': np.std(total_step_times),
            'min_step_time': np.min(total_step_times),
            'max_step_time': np.max(total_step_times),
            'total_time': np.sum(total_step_times),
            'steps_per_second': num_iterations / np.sum(total_step_times),
            'num_iterations': num_iterations
        }
    
    def benchmark_substeps(self, model_name: str, num_steps: int = 100):
        """Detailed benchmark of individual substeps"""
        if model_name not in self.simulators:
            return None
        
        print(f"Detailed substep analysis for {model_name}...")
        
        config = self.simulators[model_name]['config']
        nn_model = self.simulators[model_name]['nn_model']
        
        test_simulator = Simulator(
            l_M=self.l_M,
            penn_model=config['includes_penn'],
            damping_model=config['includes_damp'],
            nn_model=nn_model
        )
        
        test_simulator = self.instrument_simulator(test_simulator)
        
        # Collect detailed timing
        all_substep_times = []
        all_nn_times = []
        
        for i in range(num_steps):
            activation = 0.5 + 0.3 * np.sin(i * 0.1)
            
            # Clear timing arrays
            test_simulator._substep_times = []
            test_simulator._nn_forward_times = []
            
            # Run one step
            test_simulator.step(activation)
            
            # Collect timing data
            if test_simulator._substep_times:
                all_substep_times.extend(test_simulator._substep_times)
            if test_simulator._nn_forward_times:
                all_nn_times.extend(test_simulator._nn_forward_times)
        
        result = {
            'model_name': model_name,
            'substep_times': all_substep_times,
            'nn_forward_times': all_nn_times
        }
        
        if all_substep_times:
            result.update({
                'mean_substep_time': np.mean(all_substep_times),
                'std_substep_time': np.std(all_substep_times),
                'total_substeps': len(all_substep_times),
                'avg_substeps_per_step': len(all_substep_times) / num_steps
            })
        
        if all_nn_times:
            result.update({
                'mean_nn_time': np.mean(all_nn_times),
                'std_nn_time': np.std(all_nn_times),
                'total_nn_calls': len(all_nn_times)
            })
        
        return result
    
    def run_full_benchmark(self, num_iterations: int = 1000, num_runs: int = 3):
        """Run complete benchmark suite"""
        print(f"\n{'='*60}")
        print("STARTING NEURAL NETWORK BENCHMARK")
        print(f"{'='*60}")
        
        all_results = []
        substep_results = []
        
        for model_name in self.simulators.keys():
            print(f"\n--- Benchmarking {model_name} ---")
            
            # Run multiple benchmark runs for statistics
            model_runs = []
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}")
                result = self.benchmark_model(model_name, num_iterations)
                if result:
                    model_runs.append(result)
            
            if model_runs:
                # Calculate aggregate statistics
                config = model_runs[0]['config']
                mean_times = [r['mean_step_time'] for r in model_runs]
                total_times = [r['total_time'] for r in model_runs]
                steps_per_sec = [r['steps_per_second'] for r in model_runs]
                
                summary = {
                    'model_name': model_name,
                    'input_size': config['input_size'],
                    'hidden_size': config['num_width'],
                    'num_layers': config['num_layer'],
                    'includes_penn': config['includes_penn'],
                    'includes_damp': config['includes_damp'],
                    'mean_step_time_ms': np.mean(mean_times) * 1000,
                    'std_step_time_ms': np.std(mean_times) * 1000,
                    'mean_total_time_s': np.mean(total_times),
                    'steps_per_second': np.mean(steps_per_sec),
                    'std_steps_per_second': np.std(steps_per_sec),
                    'num_iterations': num_iterations,
                    'num_runs': num_runs
                }
                all_results.append(summary)
            
            # Detailed substep analysis
            substep_result = self.benchmark_substeps(model_name, num_steps=200)
            if substep_result:
                substep_results.append(substep_result)
        
        self.timing_results = {
            'summary': pd.DataFrame(all_results),
            'substeps': substep_results
        }
        
        return self.timing_results
    
    def print_results(self):
        """Print benchmark results to console"""
        if not self.timing_results:
            print("No results to display. Run benchmark first.")
            return
        
        summary_df = self.timing_results['summary']
        if summary_df.empty:
            print("No summary results available.")
            return
        
        # Sort by performance
        summary_df = summary_df.sort_values('mean_step_time_ms')
        
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Model':<30} {'Step Time (ms)':<15} {'Steps/sec':<12} {'Model Size':<15}")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            model_short = row['model_name'].replace('.pth', '').replace('_model', '')[:29]
            time_str = f"{row['mean_step_time_ms']:.2f}±{row['std_step_time_ms']:.2f}"
            steps_str = f"{row['steps_per_second']:.1f}"
            size_str = f"{row['hidden_size']}x{row['num_layers']}"
            
            print(f"{model_short:<30} {time_str:<15} {steps_str:<12} {size_str:<15}")
        
        # Performance analysis
        fastest = summary_df.iloc[0]
        slowest = summary_df.iloc[-1]
        
        print(f"\n{'='*80}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        print(f"Fastest model: {fastest['model_name']}")
        print(f"  Step time: {fastest['mean_step_time_ms']:.2f} ms")
        print(f"  Throughput: {fastest['steps_per_second']:.1f} steps/sec")
        
        print(f"\nSlowest model: {slowest['model_name']}")
        print(f"  Step time: {slowest['mean_step_time_ms']:.2f} ms")  
        print(f"  Throughput: {slowest['steps_per_second']:.1f} steps/sec")
        
        speed_ratio = slowest['mean_step_time_ms'] / fastest['mean_step_time_ms']
        print(f"\nSpeed difference: {speed_ratio:.2f}x")
        
        # Substep analysis
        if self.timing_results['substeps']:
            print(f"\n{'-'*80}")
            print("SUBSTEP TIMING ANALYSIS")
            print(f"{'-'*80}")
            
            for substep_data in self.timing_results['substeps']:
                model_name = substep_data['model_name']
                if 'mean_substep_time' in substep_data:
                    substep_ms = substep_data['mean_substep_time'] * 1000
                    print(f"{model_name:<30} Substep: {substep_ms:.3f} ms")
                    
                    if 'mean_nn_time' in substep_data:
                        nn_ms = substep_data['mean_nn_time'] * 1000
                        nn_percentage = (substep_data['mean_nn_time'] / substep_data['mean_substep_time']) * 100
                        print(f"{'':>30} NN Forward: {nn_ms:.3f} ms ({nn_percentage:.1f}% of substep)")
        
        print(f"\n{'='*80}")
    
    def plot_results(self, save_path: str = None):
        """Create visualization of benchmark results"""
        if not self.timing_results or self.timing_results['summary'].empty:
            print("No data to plot. Run benchmark first.")
            return
        
        summary_df = self.timing_results['summary'].sort_values('mean_step_time_ms')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Neural Network Model Performance Benchmark', fontsize=16, fontweight='bold')
        
        # Prepare model labels
        model_labels = [name.replace('.pth', '').replace('_model', '') for name in summary_df['model_name']]
        
        # 1. Step time comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(summary_df)), summary_df['mean_step_time_ms'], 
                       yerr=summary_df['std_step_time_ms'], capsize=5, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Mean Step Time (ms)')
        ax1.set_title('Step Time Comparison')
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(model_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, summary_df['mean_step_time_ms'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + summary_df.iloc[i]['std_step_time_ms'],
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Throughput comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(summary_df)), summary_df['steps_per_second'], 
                        alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Steps per Second')
        ax2.set_title('Throughput Comparison')
        ax2.set_xticks(range(len(summary_df)))
        ax2.set_xticklabels(model_labels, rotation=45, ha='right')
        
        for i, (bar, val) in enumerate(zip(bars2, summary_df['steps_per_second'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Model complexity vs performance
        ax3 = axes[1, 0]
        complexity = summary_df['hidden_size'] * summary_df['num_layers']
        scatter = ax3.scatter(complexity, summary_df['mean_step_time_ms'], 
                             c=summary_df['input_size'], cmap='viridis', alpha=0.8, s=100)
        ax3.set_xlabel('Model Complexity (hidden_size × layers)')
        ax3.set_ylabel('Mean Step Time (ms)')
        ax3.set_title('Performance vs Model Complexity')
        
        # Add model labels to scatter points
        for i, label in enumerate(model_labels):
            ax3.annotate(label, (complexity.iloc[i], summary_df['mean_step_time_ms'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax3, label='Input Size')
        
        # 4. Feature impact analysis
        ax4 = axes[1, 1]
        feature_comparison = []
        labels = []
        
        penn_models = summary_df[summary_df['includes_penn'] == True]
        no_penn_models = summary_df[summary_df['includes_penn'] == False]
        
        if len(penn_models) > 0:
            feature_comparison.append(penn_models['mean_step_time_ms'].mean())
            labels.append('With Penn')
        
        if len(no_penn_models) > 0:
            feature_comparison.append(no_penn_models['mean_step_time_ms'].mean())
            labels.append('Without Penn')
        
        damp_models = summary_df[summary_df['includes_damp'] == True]
        no_damp_models = summary_df[summary_df['includes_damp'] == False]
        
        if len(damp_models) > 0:
            feature_comparison.append(damp_models['mean_step_time_ms'].mean())
            labels.append('With Damp')
        
        if len(no_damp_models) > 0:
            feature_comparison.append(no_damp_models['mean_step_time_ms'].mean())
            labels.append('Without Damp')
        
        if feature_comparison:
            bars4 = ax4.bar(labels, feature_comparison, alpha=0.8, color=['orange', 'green', 'purple', 'brown'][:len(labels)])
            ax4.set_ylabel('Mean Step Time (ms)')
            ax4.set_title('Feature Impact on Performance')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars4, feature_comparison):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor feature analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Feature Impact Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = "neural_network_benchmark.csv"):
        """Export results to CSV file"""
        if not self.timing_results or self.timing_results['summary'].empty:
            print("No results to export.")
            return
        
        self.timing_results['summary'].to_csv(filename, index=False)
        print(f"Results exported to: {filename}")


def main():
    # Model configurations
    model_configs = [
        {
            'model_name': 'act_len_64by3.pth',
            'input_size': 3,
            'num_width': 64,
            'output_size': 1,
            'num_layer': 3,
            'activation_func': 'tanh',
            'includes_penn': False,
            'includes_damp': False
        },
        {
            'model_name': 'damp_64by3.pth',
            'input_size': 3,
            'num_width': 64,
            'output_size': 1,
            'num_layer': 3,
            'activation_func': 'tanh',
            'includes_penn': False,
            'includes_damp': True
        },
        {
            'model_name': 'fullForce_64by3.pth',
            'input_size': 4,
            'num_width': 64,
            'output_size': 1,
            'num_layer': 3,
            'activation_func': 'tanh',
            'includes_penn': True,
            'includes_damp': True
        }
    ]
    
    # Create and run benchmark
    benchmark = NeuralNetworkBenchmark(model_configs)
    
    # Load all models
    benchmark.load_models()
    
    if not benchmark.simulators:
        print("No models loaded successfully. Exiting.")
        return
    
    # Run benchmark
    print(f"\nStarting benchmark with {len(benchmark.simulators)} models...")
    results = benchmark.run_full_benchmark(num_iterations=500, num_runs=3)
    
    # Display results
    benchmark.print_results()
    
    # Create visualizations
    benchmark.plot_results(save_path="nn_benchmark_results.png")
    
    # Export to CSV
    benchmark.export_results("nn_benchmark_results.csv")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()