import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import datetime
import os
import gc

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Only print device info in the main process
if __name__ == "__main__":
    print(f"Using device: {device}")

# Class to convert NPZ to memory-mapped arrays (one-time preprocessing)
class NpzToMemmap:
    @staticmethod
    def convert(npz_file, output_dir, chunk_size=1000000):
        """
        Convert NPZ file to memory-mapped numpy arrays
        
        Args:
            npz_file: Path to the NPZ file
            output_dir: Directory to store the memory-mapped files
            chunk_size: Number of samples to process at once
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Converting {npz_file} to memory-mapped format...")
        data = np.load(npz_file)
        
        # Get arrays
        lMtilde = data['L']
        activation = data['A']
        muscle_force = data['F']
        
        # Get shapes
        shape = lMtilde.shape
        total_samples = shape[0] * shape[1]  # M×N
        
        # Create memory-mapped files
        lm_memmap = np.memmap(os.path.join(output_dir, 'lMtilde.npy'), 
                               dtype='float32', mode='w+', shape=(total_samples,))
        act_memmap = np.memmap(os.path.join(output_dir, 'activation.npy'),
                               dtype='float32', mode='w+', shape=(total_samples,))
        force_memmap = np.memmap(os.path.join(output_dir, 'muscle_force.npy'),
                               dtype='float32', mode='w+', shape=(total_samples,))
        
        # Process data in chunks to avoid memory issues
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            print(f"Processing chunk {i+1}/{num_chunks} [{start_idx}:{end_idx}]")
            
            # Flatten chunk of each array
            lm_flat = lMtilde.flatten()[start_idx:end_idx]
            act_flat = activation.flatten()[start_idx:end_idx]
            force_flat = muscle_force.flatten()[start_idx:end_idx]
            
            # Copy to memory-mapped arrays
            lm_memmap[start_idx:end_idx] = lm_flat
            act_memmap[start_idx:end_idx] = act_flat
            force_memmap[start_idx:end_idx] = force_flat
            
            # Flush to disk
            lm_memmap.flush()
            act_memmap.flush()
            force_memmap.flush()
            
            # Free memory
            del lm_flat, act_flat, force_flat
            gc.collect()
        
        # Save metadata about the arrays
        np.save(os.path.join(output_dir, 'metadata.npy'), {
            'total_samples': total_samples,
            'original_shape': shape,
            'lm_range': [float(np.min(lm_memmap)), float(np.max(lm_memmap))],
            'act_range': [float(np.min(act_memmap)), float(np.max(act_memmap))],
            'force_range': [float(np.min(force_memmap)), float(np.max(force_memmap))]
        })
        
        print(f"Conversion complete. Memory-mapped files saved to {output_dir}")
        
        # Close memory-mapped files
        del lm_memmap, act_memmap, force_memmap
        gc.collect()
        
        return output_dir

# Custom dataset for memory-mapped data
class MemmapMuscleDataset(Dataset):
    def __init__(self, data_dir, indices=None, cache_size=10000):
        """
        Dataset for memory-mapped muscle data
        
        Args:
            data_dir: Directory containing the memory-mapped files
            indices: Indices to use (for train/valid split)
            cache_size: Number of samples to cache in memory
        """
        self.data_dir = data_dir
        
        # Load the metadata
        try:
            self.metadata = np.load(os.path.join(data_dir, 'metadata.npy'), allow_pickle=True).item()
            self.total_samples = self.metadata['total_samples']
        except:
            # If metadata doesn't exist, try to infer from files
            lm_path = os.path.join(data_dir, 'lMtilde.npy')
            lm_shape = np.memmap(lm_path, dtype='float32', mode='r').shape
            self.total_samples = lm_shape[0]
            self.metadata = {'total_samples': self.total_samples}
        
        # Set indices (for train/valid split)
        self.indices = indices if indices is not None else np.arange(self.total_samples)
        
        # Open memory-mapped files in read-only mode
        self.lm_memmap = np.memmap(os.path.join(data_dir, 'lMtilde.npy'), 
                                    dtype='float32', mode='r')
        self.act_memmap = np.memmap(os.path.join(data_dir, 'activation.npy'),
                                    dtype='float32', mode='r')
        self.force_memmap = np.memmap(os.path.join(data_dir, 'muscle_force.npy'),
                                    dtype='float32', mode='r')
        
        # Setup caching for frequently accessed samples
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"Dataset initialized with {len(self.indices)} samples")
        if 'lm_range' in self.metadata:
            print(f"Input range - lMtilde: {self.metadata['lm_range']}, " +
                  f"activation: {self.metadata['act_range']}")
            print(f"Output range - force: {self.metadata['force_range']}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample by index with caching for frequently accessed samples"""
        actual_idx = self.indices[idx]
        
        # Check if in cache
        if actual_idx in self.cache:
            self.cache_hits += 1
            lm, act, force = self.cache[actual_idx]
        else:
            self.cache_misses += 1
            # Read from memory-mapped files
            lm = self.lm_memmap[actual_idx]
            act = self.act_memmap[actual_idx]
            force = self.force_memmap[actual_idx]
            
            # Add to cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[actual_idx] = (lm, act, force)
            
        # Create input tensor
        inputs = torch.tensor([lm, act], dtype=torch.float32)
        outputs = torch.tensor([force], dtype=torch.float32)
        
        return inputs, outputs
    
    def get_cache_stats(self):
        """Get cache hit/miss statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def reset_cache_stats(self):
        """Reset cache statistics"""
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}

# Define the model (unchanged)
class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom loss function (unchanged but adapted for current input format)
def custom_loss(model, batch):
    x, target_f = batch
    x = x.to(device)
    target_f = target_f.to(device)
    
    # Make input require gradient for computing dC/dx (wrt lMtilde)
    inputs = x.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Forward pass to get C(x)
        C_values = model(inputs)
        
        # Compute gradient dC/dx with respect to lMtilde (first input column)
        # We use create_graph=True because we need to backpropagate through this gradient
        C_grad = torch.autograd.grad(
            outputs=C_values,
            inputs=inputs,
            grad_outputs=torch.ones_like(C_values, device=device),
            create_graph=True
        )[0]
        
        # Extract only the gradient with respect to lMtilde (first component)
        lMtilde_grad = C_grad[:, 0].view(-1, 1)
        
        # Compute C(x)*dC/dx = -f(x) relationship
        left_side = C_values * lMtilde_grad  # C(x)*dC/dx
        right_side = -target_f  # -f(x)
        
        residual = left_side - right_side  # C(x)*dC/dx - (-f(x))
        
        loss = torch.mean(residual**2)
    return loss

# Function to prepare data using memory-mapped files
def prepare_data_memmap(data_dir, batch_size=128, cache_size=10000, num_workers=4):
    print(f"Preparing data from memory-mapped files in {data_dir}...")
    
    # Create the dataset
    full_dataset = MemmapMuscleDataset(data_dir, cache_size=cache_size)
    
    # Split into train and validation sets
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_dataset = MemmapMuscleDataset(data_dir, indices=train_indices, cache_size=cache_size)
    valid_dataset = MemmapMuscleDataset(data_dir, indices=valid_indices, cache_size=cache_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, valid_loader

# Train model function (largely unchanged except for cache stats reporting)
def train_model(model, train_loader, valid_loader, epochs=1000, lr=0.001, result_dir='TrainedResults', model_dir='TrainedModels'):
    # Move model to GPU
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Initialize TensorBoard writer with a descriptive name
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(result_dir, "MuscleWithActAndPennation", f"gpu_memmap_{runTime}"))

    # Training metrics
    best_valid_loss = float('inf')
    epoch_times = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model architecture:\n{model}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()
            
            # Forward pass with the custom loss
            loss = custom_loss(model, batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                loss = custom_loss(model, batch)
                valid_loss += loss.item()
                
        avg_valid_loss = valid_loss / len(valid_loader)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.8f}, '
              f'Validation Loss: {avg_valid_loss:.8f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Report cache statistics if available (every 10 epochs)
        if epoch % 10 == 0 and hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            print(f"Cache stats - hits: {cache_stats['hits']}, misses: {cache_stats['misses']}, "
                 f"hit rate: {cache_stats['hit_rate']:.2%}")
            writer.add_scalar('Cache/HitRate', cache_stats['hit_rate'], epoch)
            train_loader.dataset.reset_cache_stats()
        
        # Log the losses to TensorBoard
        writer.add_scalars('Losses', {
            'Training': avg_train_loss,
            'Validation': avg_valid_loss
        }, epoch)
        writer.add_scalar('Epoch Time', epoch_time, epoch)

        # Learning rate scheduling
        scheduler.step(avg_valid_loss)
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
            }, os.path.join(model_dir, 'gpu_muscle_memmap_best_model.pth'))
            
        # Early stopping if validation loss is extremely small
        if avg_valid_loss < 1e-6:
            print(f"Validation loss {avg_valid_loss:.10f} is very small. Early stopping.")
            break

        # Report GPU memory usage if using CUDA
        if device.type == 'cuda':
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
            writer.add_scalar('GPU/Memory_Allocated_MB', gpu_memory_allocated, epoch)
            writer.add_scalar('GPU/Memory_Reserved_MB', gpu_memory_reserved, epoch)
            
            if epoch % 10 == 0:  # Print every 10 epochs to keep logs cleaner
                print(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")

    # Print training statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Training completed in {len(epoch_times)} epochs")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    print(f"Best validation loss: {best_valid_loss:.10f}")

    # Close the TensorBoard writer
    writer.close()

    # Save the final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'valid_loss': avg_valid_loss,
    }, os.path.join(model_dir, 'gpu_muscle_memmap_final_model.pth'))
    
    return model, best_valid_loss, epoch_times

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train muscle model on GPU with memory-mapped data')
    parser.add_argument('--data', type=str, default='TrainingData/MuscleWithAct/lM_act_force_wider.npz', 
                        help='Path to the data file (NPZ format)')
    parser.add_argument('--memmap_dir', type=str, default='TrainingData/memmap_data', 
                        help='Directory to store memory-mapped files')
    parser.add_argument('--preprocess', action='store_true', 
                        help='Preprocess NPZ data to memory-mapped format')
    parser.add_argument('--chunk_size', type=int, default=1000000, 
                        help='Chunk size for preprocessing')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--cache_size', type=int, default=100000, 
                        help='Number of samples to cache in memory')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, 
                        help='Hidden layer size')
    parser.add_argument('--model_dir', type=str, default='TrainedModels', 
                        help='Output directory for models')
    parser.add_argument('--result_dir', type=str, default='TrainedResults', 
                        help='Output directory for training metrics')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "MuscleWithActAndPennation"), exist_ok=True)
    
    # Preprocess data if required
    if args.preprocess:
        data_dir = NpzToMemmap.convert(args.data, args.memmap_dir, args.chunk_size)
    else:
        data_dir = args.memmap_dir
        if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, 'lMtilde.npy')):
            print(f"Memory-mapped data not found in {data_dir}. Run with --preprocess flag first.")
            exit(1)
    
    # Prepare data
    train_loader, valid_loader = prepare_data_memmap(
        data_dir, 
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        num_workers=args.num_workers
    )
    
    # Create model with specified hidden size
    model = MLP(input_size=2, hidden_size=args.hidden_size, output_size=1)
    
    # Train model
    model, best_loss, epoch_times = train_model(
        model, 
        train_loader, 
        valid_loader, 
        epochs=args.epochs,
        lr=args.lr,
        model_dir=args.model_dir,
        result_dir=args.result_dir
    )
    
    print(f"Training complete! Best validation loss: {best_loss:.10f}")
    print(f"Models saved in {args.model_dir}")