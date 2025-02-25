import torch
import os
import tempfile
import sys
import gc

# Check if PYTORCH_MPS_HIGH_WATERMARK_RATIO is set
watermark_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')
print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {watermark_ratio}")

# Print system info
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS recommended max memory: {torch.mps.recommended_max_memory() / (1024**3):.2f} GB")
print(f"Current MPS allocated: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB")
print(f"Driver MPS allocated: {torch.mps.driver_allocated_memory() / (1024**3):.2f} GB")

# Create multiple tensors to try to exceed MPS memory
print("\n=== Testing with multiple large tensors ===")

# Create a list to hold our tensors
tensors = []

# Try to allocate tensors until we hit an error
try:
    # Each tensor will be about 2GB
    size = 23000  # ~2GB for float32
    max_tensors = 3  # Try to allocate up to 6GB (should exceed the 5.33GB limit)
    
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

# Now test with memory-mapped tensors
print("\n=== Testing with memory-mapped tensors ===")

# Create a temporary file for a large tensor
with tempfile.NamedTemporaryFile(delete=False) as f:
    temp_file = f.name
    
    # Create a smaller tensor for saving (to avoid memory issues during creation)
    size = 8000  # Smaller size for initial creation
    print(f"\nCreating tensor of size {size}x{size} for memory-mapping test")
    
    t = torch.randn(size, size)
    tensor_size_gb = t.element_size() * t.numel() / (1024**3)
    print(f'Original tensor size: {tensor_size_gb:.2f} GB')
    
    torch.save(t, temp_file)
    print(f'Saved tensor size: {os.path.getsize(temp_file) / (1024**3):.2f} GB')
    
    # Free original tensor
    del t
    gc.collect()
    torch.mps.empty_cache()

# Load with mmap=True
print("\nLoading with mmap=True:")
try:
    loaded = torch.load(temp_file, map_location='cpu', mmap=True)
    print(f'Loaded tensor shape: {loaded.shape}')
    print(f'Is memory-mapped: {getattr(loaded, "is_shared", lambda: False)()}')
    
    # Check MPS memory before moving
    print(f'MPS memory before moving: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')
    
    # Move to MPS
    print("Moving memory-mapped tensor to MPS...")
    loaded_mps = loaded.to('mps')
    print(f'MPS memory after moving: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')
    
    # Clean up
    del loaded_mps
    torch.mps.empty_cache()
    print(f'MPS memory after cleanup: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')
except Exception as e:
    print(f"Error: {e}")

# Clean up temp file
os.unlink(temp_file)

print("\n=== Testing with invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO ===")
print("This test simulates the error in the traceback")

try:
    # Try to set an invalid watermark ratio
    print("Setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4 (invalid value)")
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '1.4'
    
    # Try to create a tensor on MPS
    print("Creating a tensor on MPS with invalid watermark ratio...")
    test_tensor = torch.randn(1000, 1000).to('mps')
    print("Successfully created tensor (unexpected)")
except Exception as e:
    print(f"Error (expected): {e}")
finally:
    # Reset the environment variable
    if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
        del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    print("Reset PYTORCH_MPS_HIGH_WATERMARK_RATIO")

print("\nTest completed successfully") 