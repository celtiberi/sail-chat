import torch
import os
import gc

# Check environment
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')}")
print(f"MPS recommended max memory: {torch.mps.recommended_max_memory() / (1024**3):.2f} GB")

# Create multiple large tensors to try to exceed MPS memory
print("\nCreating multiple large tensors to exceed recommended max memory...")

# Create a list to hold our tensors
tensors = []

# Try to allocate tensors until we hit an error
try:
    # Each tensor will be about 3GB
    size = 28000  # ~3GB for float32
    max_tensors = 3  # Try to allocate up to 9GB (should exceed the 5.33GB limit)
    
    for i in range(max_tensors):
        print(f"\nAttempting to create tensor {i+1}/{max_tensors} of size {size}x{size}")
        tensor = torch.randn(size, size, device='mps')
        tensor_size_gb = tensor.element_size() * tensor.numel() / (1024**3)
        tensors.append(tensor)
        
        print(f'Created tensor {i+1} of size: {tensor_size_gb:.2f} GB')
        print(f'MPS memory now: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')
        print(f'Driver allocated: {torch.mps.driver_allocated_memory() / (1024**3):.2f} GB')
except Exception as e:
    print(f"\nHit an error: {e}")
finally:
    # Clean up
    print("\nCleaning up tensors...")
    for tensor in tensors:
        del tensor
    tensors = []
    gc.collect()
    torch.mps.empty_cache()
    print(f'MPS memory after cleanup: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')

print("\nTest completed successfully") 