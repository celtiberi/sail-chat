import torch
import os

# Set an invalid watermark ratio
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '1.4'
print(f"Set PYTORCH_MPS_HIGH_WATERMARK_RATIO to {os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}")

try:
    # Try to create a tensor on MPS
    print("Attempting to create tensor on MPS...")
    t = torch.randn(1000, 1000).to('mps')
    print("Successfully created tensor (unexpected)")
except Exception as e:
    print(f"Error (expected): {e}") 