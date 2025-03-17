import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import datetime
import os
import gc
from typing import Iterator, List, Optional
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple chunked sampler that works without multiprocessing
class SingleProcessChunkedSampler(Sampler):
    """
    Samples elements sequentially in chunks, with shuffling inside each chunk.
    Designed for single-process use to avoid pickling issues.
    """
    
    def __init__(self, data_length, chunk_size=10000000, shuffle_chunks=True):
        self.data_length = data_length
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        
        # Calculate number of chunks
        self.num_chunks = (self.data_length + self.chunk_size - 1) // self.chunk_size
        
        print(f"SingleProcessChunkedSampler: {self.data_length} samples, "
              f"{self.num_chunks} chunks of size {self.chunk_size}")
    
    def __iter__(self):
        # Create chunk indices
        chunk_indices = list(range(self.num_chunks))
        
        # Shuffle chunk order if requested
        if self.shuffle_chunks:
            chunk_indices = np.random.permutation(chunk_indices).tolist()
        
        # Process each chunk
        for chunk_idx in chunk_indices:
            # Calculate start and end indices for this chunk
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.data_length)
            
            # Create array of indices for this chunk and shuffle
            chunk_indices = np.arange(start_idx, end_idx)
            np.random.shuffle(chunk_indices)
            
            # Yield indices from this chunk
            for idx in chunk_indices:
                yield int(idx)
    
    def __len__(self):
        return self.data_length

# Simple dataset for memory-mapped data without multiprocessing
class SingleProcessMuscleDataset(Dataset):
    """
    Dataset for memory-mapped muscle data that works with a single process.
    Avoids pickling issues by keeping everything in a single process.
    """
    def __init__(self, data_dir, indices=None, cache_size=10000):
        self.data_dir = data_dir
        
        # Load memory-mapped files directly in main process
        print(f"Loading memory-mapped files from {data_dir}")
        self.lm_memmap = np.load(os.path.join(data_dir, 'lMtilde.npy'), mmap_mode='r')
        self.act_memmap = np.load(os.path.join(data_dir, 'activation.npy'), mmap_mode='r')
        self.pen_memmap = np.load(os.path.join(data_dir, 'pennation.npy'), mmap_mode='r')
        self.force_memmap = np.load(os.path.join(data_dir, 'force.npy'), mmap_mode='r')
        
        self.total_samples = len(self.lm_memmap)
        print(f"Total samples: {self.total_samples}")
        
        # Set indices (for train/valid split)
        self.indices = indices if indices is not None else np.arange(self.total_samples)
        
        # Cache parameters
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Compute data ranges from a small sample
        sample_size = min(1000, self.total_samples)
        sample_indices = np.random.choice(self.total_samples, sample_size, replace=False)
        
        lm_sample = self.lm_memmap[sample_indices]
        act_sample = self.act_memmap[sample_indices]
        pen_sample = self.pen_memmap[sample_indices]
        force_sample = self.force_memmap[sample_indices]
        
        print(f"Data ranges:")
        print(f"  lMtilde: [{np.min(lm_sample):.4f}, {np.max(lm_sample):.4f}]")
        print(f"  activation: [{np.min(act_sample):.4f}, {np.max(act_sample):.4f}]") 
        print(f"  pennation: [{np.min(pen_sample):.4f}, {np.max(pen_sample):.4f}]")
        print(f"  force: [{np.min(force_sample):.4f}, {np.max(force_sample):.4f}]")
        
        print(f"Dataset initialized with {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample by index with caching for frequently accessed samples"""
        actual_idx = self.indices[idx]
        
        # Check if in cache
        if actual_idx in self.cache:
            self.cache_hits += 1
            lm, act, pen, force = self.cache[actual_idx]
        else:
            self.cache_misses += 1
            # Read from memory-mapped files
            lm = self.lm_memmap[actual_idx]
            act = self.act_memmap[actual_idx]
            pen = self.pen_memmap[actual_idx]
            force = self.force_memmap[actual_idx]
            
            # Add to cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[actual_idx] = (lm, act, pen, force)
            # If cache is full, implement simple LRU by removing random entry
            elif self.cache and len(self.cache) >= self.cache_size:
                # Remove a key (simple approximation of LRU)
                remove_key = next(iter(self.cache.keys()))
                del self.cache[remove_key]
                self.cache[actual_idx] = (lm, act, pen, force)
            
        # Create input tensor with three inputs
        inputs = torch.tensor([lm, act, pen], dtype=torch.float32)
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
        self.cache.clear()

# Function to prepare data using memory-mapped files without multiprocessing
def prepare_single_process_data(data_dir, batch_size=128, cache_size=10000, 
                              chunk_size=10000000, shuffle_chunks=True):
    print(f"Preparing data from memory-mapped files in {data_dir} (single process mode)...")
    
    # Check files before loading
    required_files = ['lMtilde.npy', 'activation.npy', 'pennation.npy', 'force.npy']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")
        else:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"Found {file}: {file_size:.2f} MB")
    
    # Create the dataset
    print("Loading dataset...")
    full_dataset = SingleProcessMuscleDataset(data_dir, cache_size=cache_size)
    
    # Split into train and validation sets
    total_samples = len(full_dataset)
    indices = np.arange(total_samples)
    
    print("Splitting into train and validation sets...")
    valid_size = min(int(0.2 * total_samples), 1000000)  # Cap validation size
    train_size = total_samples - valid_size
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    # Create train and validation datasets
    train_dataset = SingleProcessMuscleDataset(data_dir, indices=train_indices, cache_size=cache_size)
    valid_dataset = SingleProcessMuscleDataset(data_dir, indices=valid_indices, cache_size=cache_size)
    
    # Create samplers
    train_sampler = SingleProcessChunkedSampler(
        len(train_dataset),
        chunk_size=min(chunk_size, len(train_dataset)),
        shuffle_chunks=shuffle_chunks
    )
    
    valid_sampler = SingleProcessChunkedSampler(
        len(valid_dataset),
        chunk_size=min(1000000, len(valid_dataset)),
        shuffle_chunks=False
    )
    
    # Create dataloaders - note: num_workers=0 to avoid multiprocessing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # Single process
        pin_memory=True if device.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,  # Single process
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, valid_loader

# Define the model with three inputs (unchanged)
class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=1):
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

# Custom loss function (unchanged)
def custom_loss(model, batch):
    x, target_f = batch
    x = x.to(device)
    target_f = target_f.to(device)
    
    # Make input require gradient for computing dC/dx (wrt lMtilde)
    inputs = x.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Forward pass to get C(x)
        C_values = model(inputs)
        
        # Compute gradient dC/dx with respect to lMtilde (first input component)
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

# Train model function (main parts unchanged, adjusted for simpler progress reporting)
def train_model(model, train_loader, valid_loader, epochs=1000, lr=0.001, 
               result_dir='TrainedResults', model_dir='TrainedModels'):
    # Move model to GPU
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Initialize TensorBoard writer
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(result_dir, "MuscleWithActAndPennation", f"single_process_{runTime}"))

    # Training metrics
    best_valid_loss = float('inf')
    epoch_times = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model architecture:\n{model}")
    
    # Create a progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training progress", unit="epoch")
    
    for epoch in epoch_pbar:
        start_time = time.time()
            
        # Update epoch progress bar description
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                          leave=False, unit="batch")
        
        for batch in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with the custom loss
            loss = custom_loss(model, batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.8f}"})
            
        avg_train_loss = train_loss / batch_count

        # Validation phase
        model.eval()
        valid_loss = 0
        valid_batch_count = 0
        
        # Create progress bar for validation
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", 
                         leave=False, unit="batch")
        
        with torch.no_grad():
            for batch in valid_pbar:
                loss = custom_loss(model, batch)
                valid_loss += loss.item()
                valid_batch_count += 1
                
                # Update progress bar with current loss
                valid_pbar.set_postfix({"loss": f"{loss.item():.8f}"})
                
        avg_valid_loss = valid_loss / valid_batch_count
        
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        # Update epoch progress bar with results
        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.8f}",
            "valid_loss": f"{avg_valid_loss:.8f}",
            "time": f"{epoch_time:.2f}s"
        })
        
        # Report cache statistics from the datasets
        try:
            train_cache_stats = train_loader.dataset.get_cache_stats()
            print(f"Train cache stats - hits: {train_cache_stats['hits']}, misses: {train_cache_stats['misses']}, "
                 f"hit rate: {train_cache_stats['hit_rate']:.2%}")
            writer.add_scalar('Cache/Train_HitRate', train_cache_stats['hit_rate'], epoch)
            
            valid_cache_stats = valid_loader.dataset.get_cache_stats()
            print(f"Valid cache stats - hits: {valid_cache_stats['hits']}, misses: {valid_cache_stats['misses']}, "
                 f"hit rate: {valid_cache_stats['hit_rate']:.2%}")
            writer.add_scalar('Cache/Valid_HitRate', valid_cache_stats['hit_rate'], epoch)
        except Exception as e:
            print(f"Error getting cache stats: {e}")
        
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
            }, os.path.join(model_dir, 'muscle_3inputs_best_model.pth'))
            
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
    }, os.path.join(model_dir, 'muscle_3inputs_final_model.pth'))
    
    return model, best_valid_loss, epoch_times

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train muscle model on GPU with single-process memory-mapped data')
    parser.add_argument('--data_dir', type=str, default='TrainingData/muscle_force_3Dinputs', 
                        help='Directory containing NPY files')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--cache_size', type=int, default=100000, 
                        help='Number of samples to cache in memory')
    parser.add_argument('--chunk_size', type=int, default=1000000, 
                        help='Number of samples per chunk for sequential access')
    parser.add_argument('--no_shuffle_chunks', action='store_false', dest='shuffle_chunks',
                        help='Disable shuffling of chunk order')
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
    
    # Set defaults
    parser.set_defaults(shuffle_chunks=True)
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "MuscleWithActAndPennation"), exist_ok=True)
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cache size: {args.cache_size}")
    print(f"Mode: Single process (no multiprocessing)")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Shuffle chunks: {args.shuffle_chunks}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Model directory: {args.model_dir}")
    print(f"Result directory: {args.result_dir}")
    print("===================\n")
    
    # Prepare data
    train_loader, valid_loader = prepare_single_process_data(
        args.data_dir, 
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        chunk_size=args.chunk_size,
        shuffle_chunks=args.shuffle_chunks
    )
    
    # Create model with specified hidden size
    model = MLP(input_size=3, hidden_size=args.hidden_size, output_size=1)
    
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