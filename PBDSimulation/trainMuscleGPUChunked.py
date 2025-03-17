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
# Only print device info in the main process
if __name__ == "__main__":
    print(f"Using device: {device}")

# Custom Chunked Sequential Sampler with local shuffling
class ChunkedSequentialSampler(Sampler):
    """
    Samples elements sequentially in chunks, with shuffling inside each chunk.
    
    Arguments:
        data_source: Dataset to sample from
        chunk_size: Number of samples per chunk
        shuffle_chunks: Whether to shuffle the order of chunks each epoch
    """
    
    def __init__(self, data_source: Dataset, chunk_size: int = 10000000, shuffle_chunks: bool = True):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        
        # Calculate number of chunks
        self.num_samples = len(self.data_source)
        self.num_chunks = (self.num_samples + self.chunk_size - 1) // self.chunk_size
        
        print(f"ChunkedSequentialSampler: {self.num_samples} samples, "
              f"{self.num_chunks} chunks of size {self.chunk_size}")
    
    def __iter__(self) -> Iterator[int]:
        # Create chunk indices
        chunk_indices = list(range(self.num_chunks))
        
        # Shuffle chunk order if requested
        if self.shuffle_chunks:
            np.random.shuffle(chunk_indices)
        
        # Process each chunk
        for chunk_idx in chunk_indices:
            # Calculate start and end indices for this chunk
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.num_samples)
            
            # Create array of indices for this chunk
            chunk_sample_indices = np.arange(start_idx, end_idx)
            
            # Shuffle indices within the chunk
            np.random.shuffle(chunk_sample_indices)
            
            # Yield indices from this chunk
            for idx in chunk_sample_indices:
                yield idx
    
    def __len__(self) -> int:
        return self.num_samples

# Custom dataset for memory-mapped data with three inputs
class MemmapMuscleDataset(Dataset):
    def __init__(self, data_dir, indices=None, cache_size=10000):
        """
        Dataset for memory-mapped muscle data with three inputs
        
        Args:
            data_dir: Directory containing the memory-mapped files
            indices: Indices to use (for train/valid split)
            cache_size: Number of samples to cache in memory
        """
        self.data_dir = data_dir
        
        # Open memory-mapped files in read-only mode
        self.lm_memmap = np.memmap(os.path.join(data_dir, 'lMtilde.npy'), 
                                   dtype='float32', mode='r')
        self.act_memmap = np.memmap(os.path.join(data_dir, 'activation.npy'),
                                   dtype='float32', mode='r')
        self.pen_memmap = np.memmap(os.path.join(data_dir, 'pennation.npy'),
                                   dtype='float32', mode='r')
        self.force_memmap = np.memmap(os.path.join(data_dir, 'force.npy'),
                                     dtype='float32', mode='r')
        
        # Total samples is determined by the shape of any of the arrays
        self.total_samples = len(self.lm_memmap)
        
        # Set indices (for train/valid split)
        self.indices = indices if indices is not None else np.arange(self.total_samples)
        
        # Setup caching for frequently accessed samples
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize cache statistics
        self.chunk_hits = {}  # Track hits by chunk
        self.last_chunk = -1
        self.sequential_hits = 0
        self.random_hits = 0
        
        # Compute data ranges for logging
        # Using small sample to avoid loading entire dataset
        sample_size = min(1000, self.total_samples)
        sample_indices = np.random.choice(self.total_samples, sample_size, replace=False)
        
        lm_sample = self.lm_memmap[sample_indices]
        act_sample = self.act_memmap[sample_indices]
        pen_sample = self.pen_memmap[sample_indices]
        force_sample = self.force_memmap[sample_indices]
        
        self.lm_range = [float(np.min(lm_sample)), float(np.max(lm_sample))]
        self.act_range = [float(np.min(act_sample)), float(np.max(act_sample))]
        self.pen_range = [float(np.min(pen_sample)), float(np.max(pen_sample))]
        self.force_range = [float(np.min(force_sample)), float(np.max(force_sample))]
        
        print(f"Dataset initialized with {len(self.indices)} samples")
        print(f"Input ranges - lMtilde: {self.lm_range}, " +
              f"activation: {self.act_range}, " +
              f"pennation: {self.pen_range}")
        print(f"Output range - force: {self.force_range}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample by index with caching for frequently accessed samples"""
        actual_idx = self.indices[idx]
        
        # Track sequential vs. random access pattern
        chunk_idx = actual_idx // 10000
        if chunk_idx == self.last_chunk:
            self.sequential_hits += 1
        else:
            self.random_hits += 1
            self.last_chunk = chunk_idx
        
        # Track hits by chunk
        if chunk_idx not in self.chunk_hits:
            self.chunk_hits[chunk_idx] = 0
        self.chunk_hits[chunk_idx] += 1
        
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
            # If cache is full, implement LRU by removing random entry
            # For simplicity - a true LRU would track access times
            elif len(self.cache) >= self.cache_size:
                # Remove a random key (simple approximation of LRU)
                remove_key = list(self.cache.keys())[0]
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
        
        sequential_total = self.sequential_hits + self.random_hits
        sequential_rate = self.sequential_hits / sequential_total if sequential_total > 0 else 0
        
        # Calculate distribution of hits across chunks
        num_active_chunks = len(self.chunk_hits)
        max_chunk_hits = max(self.chunk_hits.values()) if self.chunk_hits else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'sequential_rate': sequential_rate,
            'active_chunks': num_active_chunks,
            'max_chunk_hits': max_chunk_hits
        }
    
    def reset_cache_stats(self):
        """Reset cache statistics"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.sequential_hits = 0
        self.random_hits = 0
        self.chunk_hits = {}
        self.last_chunk = -1
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}

# Define the model with three inputs
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

# Custom loss function (updated for three inputs)
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

# Function to prepare data using memory-mapped files and chunked sampling
def prepare_data_chunked(data_dir, batch_size=128, cache_size=10000, num_workers=4, 
                        chunk_size=10000000, shuffle_chunks=True):
    print(f"Preparing data from memory-mapped files in {data_dir} with chunked sampling...")
    
    # Create the dataset with progress indication
    print("Loading dataset...")
    full_dataset = MemmapMuscleDataset(data_dir, cache_size=cache_size)
    
    # Split into train and validation sets
    total_samples = len(full_dataset)
    indices = np.arange(total_samples)
    
    print("Splitting into train and validation sets...")
    # For validation, we'll use a contiguous chunk from the end
    # This is more efficient for memory-mapped access
    valid_size = min(int(0.2 * total_samples), 1000000)  # Cap validation size for very large datasets
    train_size = total_samples - valid_size
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_dataset = MemmapMuscleDataset(data_dir, indices=train_indices, cache_size=cache_size)
    valid_dataset = MemmapMuscleDataset(data_dir, indices=valid_indices, cache_size=cache_size)
    
    # Create chunked sampler for training
    train_sampler = ChunkedSequentialSampler(
        train_dataset, 
        chunk_size=min(chunk_size, len(train_dataset)),
        shuffle_chunks=shuffle_chunks
    )
    
    # For validation, we use a smaller chunk size and sequential access is fine
    valid_sampler = ChunkedSequentialSampler(
        valid_dataset,
        chunk_size=min(1000000, len(valid_dataset)),
        shuffle_chunks=False
    )
    
    # Create dataloaders with custom samplers
    # Setting num_workers=0 to disable multiprocessing and avoid pickling errors
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # Disable multiprocessing
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,  # Disable multiprocessing
        pin_memory=True
    )
    
    return train_loader, valid_loader

# Train model function
def train_model(model, train_loader, valid_loader, epochs=1000, lr=0.001, 
               result_dir='TrainedResults', model_dir='TrainedModels'):
    # Move model to GPU
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Initialize TensorBoard writer with a descriptive name
    runTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(result_dir, "MuscleWithActAndPennation", f"gpu_chunked_{runTime}"))

    # Training metrics
    best_valid_loss = float('inf')
    epoch_times = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model architecture:\n{model}")
    
    # Create a progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training progress", unit="epoch")
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        # Reset cache stats at the beginning of each epoch
        if hasattr(train_loader.dataset, 'reset_cache_stats'):
            train_loader.dataset.reset_cache_stats()
            
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
            optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()
            
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
        
        # Report cache statistics if available
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            print(f"Cache stats - hits: {cache_stats['hits']}, misses: {cache_stats['misses']}, "
                 f"hit rate: {cache_stats['hit_rate']:.2%}, sequential: {cache_stats['sequential_rate']:.2%}")
            writer.add_scalar('Cache/HitRate', cache_stats['hit_rate'], epoch)
            writer.add_scalar('Cache/SequentialRate', cache_stats['sequential_rate'], epoch)
            writer.add_scalar('Cache/ActiveChunks', cache_stats['active_chunks'], epoch)
        
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
    parser = argparse.ArgumentParser(description='Train muscle model on GPU with chunked memory-mapped data')
    parser.add_argument('--data_dir', type=str, default='TrainingData/muscle_force_3Dinputs', 
                        help='Directory containing NPY files')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--cache_size', type=int, default=100000, 
                        help='Number of samples to cache in memory')
    parser.add_argument('--num_workers', type=int, default=0,  # Default to 0 to disable multiprocessing
                        help='Number of data loading workers (0 disables multiprocessing)')
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
    print(f"Multiprocessing: {'Disabled (num_workers=0)' if args.num_workers == 0 else f'Enabled (num_workers={args.num_workers})'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Shuffle chunks: {args.shuffle_chunks}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Model directory: {args.model_dir}")
    print(f"Result directory: {args.result_dir}")
    print("===================\n")
    
    # Prepare data
    train_loader, valid_loader = prepare_data_chunked(
        args.data_dir, 
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        num_workers=args.num_workers,
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