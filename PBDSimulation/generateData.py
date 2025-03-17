import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import argparse
import mmap
import struct
from pathlib import Path
import pandas as pd
import seaborn as sns

def load_binary_data_for_ml(filename, sample_rate=1.0, seed=42):
    """
    Load data from the binary file in a format ready for ML training.
    Uses memory mapping for efficiency and supports sampling for large datasets.
    
    Args:
        filename: Path to the binary data file
        sample_rate: Fraction of data to sample (0.0-1.0)
        seed: Random seed for reproducible sampling
        
    Returns:
        inputs: Tensor of shape (num_samples, 3) containing [lMtilde, activation, pennation]
        outputs: Tensor of shape (num_samples, 1) containing muscle forces
        metadata: Dictionary with data statistics and dimensions
    """
    start_time = time.time()
    print(f"Loading data from {filename}...")
    
    # Get file size
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")
    
    # Open the file and read the header
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(7).decode('ascii')
        if header != "NPZDATA":
            raise ValueError(f"Invalid file format: expected 'NPZDATA', got '{header}'")
        
        # Read dimensions
        dim1 = struct.unpack('I', f.read(4))[0]  # lM samples
        dim2 = struct.unpack('I', f.read(4))[0]  # activation samples
        dim3 = struct.unpack('I', f.read(4))[0]  # pennation samples
    
    print(f"Data dimensions: {dim1} (lM) x {dim2} (act) x {dim3} (pen)")
    total_samples = dim1 * dim2 * dim3
    print(f"Total data points: {total_samples:,}")
    
    # Determine how many samples to take based on the sample rate
    num_samples = int(total_samples * sample_rate)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # If sampling, generate random indices
    if sample_rate < 1.0:
        print(f"Sampling {sample_rate*100:.1f}% of data ({num_samples:,} points)...")
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        sample_indices.sort()  # Sort for sequential access
    else:
        print("Loading all data points...")
        sample_indices = None
    
    # Calculate header size and element size
    header_size = 7 + 3 * 4  # 7 for "NPZDATA" + 3 integers for dimensions
    double_size = 8  # Size of a double in bytes
    
    # Calculate total elements and each array's size
    array_size = total_samples * double_size
    
    # Prepare arrays for the data
    if sample_indices is not None:
        inputs = np.zeros((num_samples, 3), dtype=np.float32)
        outputs = np.zeros((num_samples, 1), dtype=np.float32)
    else:
        # For full dataset, we'll load arrays sequentially to save memory
        inputs = np.zeros((total_samples, 3), dtype=np.float32)
        outputs = np.zeros((total_samples, 1), dtype=np.float32)
    
    # Memory-map the file for more efficient reading
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Helper function to calculate flattened index
        def get_index(i, j, k):
            return (i * dim2 * dim3) + (j * dim3) + k
        
        # Helper function to read a double at a specific position
        def read_double(position):
            mm.seek(position)
            return struct.unpack('d', mm.read(double_size))[0]
        
        print("Extracting data...")
        load_start = time.time()
        
        if sample_indices is not None:
            # Sampled loading - load only the sampled indices
            for idx, flat_idx in enumerate(sample_indices):
                # Convert flat index back to 3D coordinates
                i = flat_idx // (dim2 * dim3)
                j = (flat_idx % (dim2 * dim3)) // dim3
                k = flat_idx % dim3
                
                # Extract the data for this sample
                # L value
                pos = header_size + (get_index(i, j, k) * double_size)
                inputs[idx, 0] = read_double(pos)
                
                # A value
                pos = header_size + array_size + (get_index(i, j, k) * double_size)
                inputs[idx, 1] = read_double(pos)
                
                # P value
                pos = header_size + 2 * array_size + (get_index(i, j, k) * double_size)
                inputs[idx, 2] = read_double(pos)
                
                # F value
                pos = header_size + 3 * array_size + (get_index(i, j, k) * double_size)
                outputs[idx, 0] = read_double(pos)
                
                # Print progress
                if idx % (num_samples // 10) == 0:
                    print(f"Progress: {idx/num_samples*100:.1f}%")
        else:
            # Full dataset loading - more efficient sequential access
            # For each input dimension, load all values sequentially
            print("Loading lMtilde values...")
            for idx in range(total_samples):
                pos = header_size + (idx * double_size)
                inputs[idx, 0] = read_double(pos)
                
                if idx % (total_samples // 10) == 0:
                    print(f"Progress: {idx/total_samples*100:.1f}%")
            
            print("Loading activation values...")
            for idx in range(total_samples):
                pos = header_size + array_size + (idx * double_size)
                inputs[idx, 1] = read_double(pos)
                
                if idx % (total_samples // 10) == 0:
                    print(f"Progress: {idx/total_samples*100:.1f}%")
            
            print("Loading pennation values...")
            for idx in range(total_samples):
                pos = header_size + 2 * array_size + (idx * double_size)
                inputs[idx, 2] = read_double(pos)
                
                if idx % (total_samples // 10) == 0:
                    print(f"Progress: {idx/total_samples*100:.1f}%")
            
            print("Loading force values...")
            for idx in range(total_samples):
                pos = header_size + 3 * array_size + (idx * double_size)
                outputs[idx, 0] = read_double(pos)
                
                if idx % (total_samples // 10) == 0:
                    print(f"Progress: {idx/total_samples*100:.1f}%")
        
        # Close memory map
        mm.close()
    
    load_end = time.time()
    print(f"Data extraction completed in {load_end - load_start:.2f} seconds")
    
    # Calculate metadata and statistics
    metadata = {
        'dims': (dim1, dim2, dim3),
        'total_samples': total_samples,
        'loaded_samples': inputs.shape[0],
        'lM_range': (float(np.min(inputs[:, 0])), float(np.max(inputs[:, 0]))),
        'act_range': (float(np.min(inputs[:, 1])), float(np.max(inputs[:, 1]))),
        'pen_range': (float(np.min(inputs[:, 2])), float(np.max(inputs[:, 2]))),
        'force_range': (float(np.min(outputs)), float(np.max(outputs))),
        'lM_mean': float(np.mean(inputs[:, 0])),
        'lM_std': float(np.std(inputs[:, 0])),
        'file_size_gb': file_size / (1024*1024*1024)
    }
    
    print(f"Input ranges:")
    print(f"  lMtilde: [{metadata['lM_range'][0]:.4f}, {metadata['lM_range'][1]:.4f}]")
    print(f"  activation: [{metadata['act_range'][0]:.4f}, {metadata['act_range'][1]:.4f}]")
    print(f"  pennation: [{metadata['pen_range'][0]:.4f}, {metadata['pen_range'][1]:.4f}]")
    print(f"Output range:")
    print(f"  force: [{metadata['force_range'][0]:.4f}, {metadata['force_range'][1]:.4f}]")
    
    end_time = time.time()
    print(f"Total loading time: {end_time - start_time:.2f} seconds")
    
    return inputs, outputs, metadata

def save_npz_for_ml(inputs, outputs, metadata, output_file="muscle_data_ml.npz"):
    """
    Save the extracted data as a compact NPZ file for ML training
    """
    # Reshape the data to match the expected format for the ML training code
    nsamples = inputs.shape[0]
    dim1, dim2, dim3 = metadata['dims']
    
    # Create meshgrid arrays of the right shape but only store the sampled points
    # This is a bit of a hack but it should work with the existing ML code
    L = inputs[:, 0].reshape(nsamples, 1)
    A = inputs[:, 1].reshape(nsamples, 1)
    P = inputs[:, 2].reshape(nsamples, 1)
    F = outputs.reshape(nsamples, 1)
    
    # Save the arrays
    np.savez(output_file, L=L, A=A, P=P, F=F)
    print(f"Data saved to {output_file}")
    
    # Also save metadata as JSON for reference
    import json
    with open(output_file.replace('.npz', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def create_pytorch_dataloaders(inputs, outputs, batch_size=1024, train_ratio=0.8, 
                              device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create PyTorch DataLoaders for training and validation
    """
    print(f"Creating PyTorch DataLoaders with batch size {batch_size}")
    print(f"Using device: {device}")
    
    # Convert to tensor format and move to device
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32, device=device)
    
    # Combine inputs and outputs into the dataset
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    
    # Create train/validation split
    train, valid = torch.utils.data.random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=(device=='cuda'))
    valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=(device=='cuda'))
    
    print(f"Created train loader with {train_size:,} samples and valid loader with {valid_size:,} samples")
    
    return train_loader, valid_loader

def visualize_data_distribution(inputs, outputs, metadata, output_dir="visualizations"):
    """
    Create visualizations of the data distribution for better understanding
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Create a pandas DataFrame for easier plotting
    df = pd.DataFrame({
        'lMtilde': inputs[:, 0],
        'activation': inputs[:, 1],
        'pennation': inputs[:, 2],
        'force': outputs[:, 0]
    })
    
    # 1. Histograms of input distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(df['lMtilde'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of lMtilde')
    axes[0, 0].set_xlabel('Normalized Muscle Length')
    
    sns.histplot(df['activation'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Activation')
    axes[0, 1].set_xlabel('Activation')
    
    sns.histplot(df['pennation'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Pennation')
    axes[1, 0].set_xlabel('Pennation Angle')
    
    sns.histplot(df['force'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Force')
    axes[1, 1].set_xlabel('Force')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distributions.png")
    plt.close()
    
    # 2. Sample scatter plots to show relationships
    # Sample 10,000 points to avoid overcrowding
    sample_size = min(10000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    # Create pairplot
    sns.pairplot(df_sample, vars=['lMtilde', 'activation', 'pennation', 'force'])
    plt.savefig(f"{output_dir}/pairwise_relationships.png")
    plt.close()
    
    # 3. 3D scatter plot to show the input space
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by force value
    scatter = ax.scatter(
        df_sample['lMtilde'],
        df_sample['activation'],
        df_sample['pennation'],
        c=df_sample['force'],
        cmap='viridis',
        alpha=0.7,
        s=10
    )
    
    ax.set_xlabel('lMtilde')
    ax.set_ylabel('activation')
    ax.set_zlabel('pennation')
    plt.colorbar(scatter, label='Force')
    plt.title('3D Distribution of Inputs Colored by Force')
    
    plt.savefig(f"{output_dir}/3d_input_space.png")
    plt.close()
    
    # 4. Slices through the 3D space
    # Create a series of 2D heatmaps for fixed values of one variable
    
    # Function to create slice plots
    def create_slice_plots(df, x_var, y_var, z_var, slice_var, n_slices=4):
        # Get unique slice positions evenly spaced
        slice_values = np.linspace(
            df[slice_var].min(),
            df[slice_var].max(),
            n_slices+2
        )[1:-1]  # Skip exact min/max
        
        fig, axes = plt.subplots(1, n_slices, figsize=(15, 4))
        
        for i, slice_val in enumerate(slice_values):
            # Get data points near the slice value
            epsilon = (df[slice_var].max() - df[slice_var].min()) / 20
            slice_data = df[(df[slice_var] >= slice_val - epsilon) & 
                            (df[slice_var] <= slice_val + epsilon)]
            
            if len(slice_data) > 100:  # Only plot if we have enough points
                sns.scatterplot(
                    data=slice_data,
                    x=x_var,
                    y=y_var,
                    hue=z_var,
                    palette="viridis",
                    ax=axes[i]
                )
                
                axes[i].set_title(f"{slice_var}={slice_val:.2f}")
                
        plt.tight_layout()
        return fig
    
    # Create slice plots for each variable
    lm_slices = create_slice_plots(df_sample, 'activation', 'pennation', 'force', 'lMtilde')
    lm_slices.suptitle('Force vs. Activation and Pennation at Different Muscle Lengths')
    plt.savefig(f"{output_dir}/lm_slices.png")
    plt.close()
    
    act_slices = create_slice_plots(df_sample, 'lMtilde', 'pennation', 'force', 'activation')
    act_slices.suptitle('Force vs. Length and Pennation at Different Activation Levels')
    plt.savefig(f"{output_dir}/act_slices.png")
    plt.close()
    
    pen_slices = create_slice_plots(df_sample, 'lMtilde', 'activation', 'force', 'pennation')
    pen_slices.suptitle('Force vs. Length and Activation at Different Pennation Angles')
    plt.savefig(f"{output_dir}/pen_slices.png")
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Load and prepare muscle force data for ML training')
    parser.add_argument('--input', default='lM_act_pen_force.bin', help='Input binary data file')
    parser.add_argument('--output', default='muscle_data_ml.npz', help='Output NPZ file for ML')
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of data to sample (0.0-1.0)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for dataloaders')
    parser.add_argument('--visualize', action='store_true', help='Generate data visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling and splits')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Load the data
    inputs, outputs, metadata = load_binary_data_for_ml(args.input, sample_rate=args.sample, seed=args.seed)
    
    # Save as NPZ
    save_npz_for_ml(inputs, outputs, metadata, output_file=args.output)
    
    # Create PyTorch dataloaders
    train_loader, valid_loader = create_pytorch_dataloaders(
        inputs, outputs, batch_size=args.batch_size
    )
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_data_distribution(inputs, outputs, metadata)
    
    print("Processing complete!")
    print(f"For ML training, load the data from: {args.output}")

if __name__ == "__main__":
    main()