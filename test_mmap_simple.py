import torch
import os
import tempfile
import psutil

# Check environment
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')}")
print(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# Memory check function
def check_memory():
    vm = psutil.virtual_memory()
    print(f'System memory usage: {vm.percent}%')
    print(f'Available memory: {vm.available / (1024**3):.2f} GB')
    if torch.backends.mps.is_available():
        print(f'MPS memory: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB')

# Create a temporary file with a larger tensor
with tempfile.NamedTemporaryFile(delete=False) as f:
    temp_file = f.name
    
    # Create and save a tensor
    size = 12000  # ~550MB for float32
    print(f"\nCreating tensor of size {size}x{size}")
    check_memory()
    t = torch.randn(size, size)
    tensor_size_gb = t.element_size() * t.numel() / (1024**3)
    print(f'Original tensor size: {tensor_size_gb:.2f} GB')
    check_memory()
    
    torch.save(t, temp_file)
    print(f'Saved tensor size: {os.path.getsize(temp_file) / (1024**3):.2f} GB')
    
    # Free original tensor
    del t
    check_memory()

# Test 1: Load with mmap=True, map_location='cpu'
print("\nTest 1: Loading with mmap=True, map_location='cpu'")
check_memory()
loaded_cpu = torch.load(temp_file, map_location='cpu', mmap=True)
print(f'Loaded tensor shape: {loaded_cpu.shape}')
print(f'Is memory-mapped: {getattr(loaded_cpu, "is_shared", lambda: False)()}')
print(f'Device: {loaded_cpu.device}')
check_memory()

# Test 2: Load with mmap=True, map_location='mps'
print("\nTest 2: Loading with mmap=True, map_location='mps'")
try:
    check_memory()
    loaded_mps_direct = torch.load(temp_file, map_location='mps', mmap=True)
    print(f'Loaded tensor shape: {loaded_mps_direct.shape}')
    print(f'Is memory-mapped: {getattr(loaded_mps_direct, "is_shared", lambda: False)()}')
    print(f'Device: {loaded_mps_direct.device}')
    check_memory()
except Exception as e:
    print(f"Error loading directly to MPS: {e}")

# Test 3: Load with mmap=True to CPU, then move to MPS
print("\nTest 3: Loading with mmap=True to CPU, then moving to MPS")
try:
    # Move to MPS
    check_memory()
    loaded_mps = loaded_cpu.to('mps')
    print(f'Device after moving: {loaded_mps.device}')
    check_memory()
    
    # Clean up
    del loaded_mps
    torch.mps.empty_cache()
    check_memory()
except Exception as e:
    print(f"Error moving to MPS: {e}")

# Test 4: Load with mmap=False, map_location='mps'
print("\nTest 4: Loading with mmap=False, map_location='mps'")
try:
    check_memory()
    loaded_mps_no_mmap = torch.load(temp_file, map_location='mps', mmap=False)
    print(f'Loaded tensor shape: {loaded_mps_no_mmap.shape}')
    print(f'Device: {loaded_mps_no_mmap.device}')
    check_memory()
    
    # Clean up
    del loaded_mps_no_mmap
    torch.mps.empty_cache()
    check_memory()
except Exception as e:
    print(f"Error loading directly to MPS without mmap: {e}")

# Test 5: Create a large tensor in memory and check memory usage
print("\nTest 5: Create a large tensor in memory and check memory usage")
check_memory()
large_tensor = torch.randn(size, size)
print(f'Created tensor of size: {large_tensor.element_size() * large_tensor.numel() / (1024**3):.2f} GB')
check_memory()
del large_tensor
check_memory()

# Clean up
os.unlink(temp_file)
print("\nTest completed successfully")
