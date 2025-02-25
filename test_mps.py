import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test CPU tensor creation
print("\nCreating CPU tensor...")
cpu_tensor = torch.randn(1000, 1000)
print(f"CPU tensor shape: {cpu_tensor.shape}")

# Test MPS tensor creation
print("\nCreating MPS tensor...")
try:
    mps_tensor = torch.randn(1000, 1000, device='mps')
    print(f"MPS tensor shape: {mps_tensor.shape}")
except Exception as e:
    print(f"Error creating MPS tensor: {e}")

# Test with PYTORCH_MPS_HIGH_WATERMARK_RATIO set
print("\nTesting with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
try:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO set to: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
    mps_tensor_with_ratio = torch.randn(1000, 1000, device='mps')
    print(f"MPS tensor shape with ratio set: {mps_tensor_with_ratio.shape}")
except Exception as e:
    print(f"Error creating MPS tensor with ratio set: {e}")

# Test with invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO
print("\nTesting with invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4")
try:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.4"
    print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO set to: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
    mps_tensor_invalid_ratio = torch.randn(1000, 1000, device='mps')
    print(f"MPS tensor shape with invalid ratio: {mps_tensor_invalid_ratio.shape}")
except Exception as e:
    print(f"Error creating MPS tensor with invalid ratio: {e}")

print("\nTest completed") 