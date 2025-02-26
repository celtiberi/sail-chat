#!/usr/bin/env python3
"""
Environment checker for the Sailing Assistant application.
This script checks the system environment and recommends the appropriate Docker configuration.
"""

import platform
import subprocess
import sys
import os
import shutil

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def check_platform():
    """Check the platform and return system, machine, and whether it's Apple Silicon."""
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == "Darwin" and machine == "arm64"
    
    print(f"System: {system}")
    print(f"Machine: {machine}")
    
    if is_apple_silicon:
        print("✅ Detected Apple Silicon (M1/M2)")
    elif system == "Darwin":
        print("✅ Detected macOS (Intel)")
    elif system == "Linux":
        print("✅ Detected Linux")
    elif system == "Windows":
        print("✅ Detected Windows")
    else:
        print(f"⚠️ Detected unknown system: {system} {machine}")
    
    return system, machine, is_apple_silicon

def check_docker():
    """Check if Docker is installed and running."""
    docker_path = shutil.which("docker")
    
    if not docker_path:
        print("❌ Docker not found. Please install Docker.")
        return False
    
    print(f"✅ Docker found at: {docker_path}")
    
    try:
        result = subprocess.run(
            ["docker", "info"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Docker is installed but not running or has permission issues.")
            print(f"Error: {result.stderr.strip()}")
            return False
        
        print("✅ Docker is running")
        return True
    
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def check_gpu():
    """Check for NVIDIA GPU and CUDA."""
    has_nvidia = False
    has_cuda = False
    
    # Check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                has_nvidia = True
                print("✅ NVIDIA GPU detected")
                
                # Extract driver version
                for line in result.stdout.split('\n'):
                    if "Driver Version" in line:
                        print(f"   {line.strip()}")
            else:
                print("❌ nvidia-smi found but returned an error")
                print(f"   Error: {result.stderr.strip()}")
        
        except Exception as e:
            print(f"❌ Error running nvidia-smi: {e}")
    else:
        print("❌ No NVIDIA GPU detected (nvidia-smi not found)")
    
    # Check for CUDA
    try:
        # Try to import torch and check CUDA availability
        import torch
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            cuda_version = torch.version.cuda
            print(f"✅ CUDA is available (version {cuda_version})")
            print(f"   Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA is not available in PyTorch")
    
    except ImportError:
        print("❌ PyTorch not installed, cannot check CUDA availability")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    return has_nvidia, has_cuda

def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        
        if "nvidia" in result.stdout:
            print("✅ NVIDIA Docker runtime is available")
            return True
        else:
            print("❌ NVIDIA Docker runtime not found")
            return False
    
    except Exception as e:
        print(f"❌ Error checking NVIDIA Docker runtime: {e}")
        return False

def recommend_configuration(is_apple_silicon, has_nvidia, has_cuda):
    """Recommend the appropriate Docker configuration."""
    print_header("Recommended Configuration")
    
    if has_nvidia and has_cuda:
        print("🚀 Recommended: GPU Configuration")
        print("\nRun the application with:")
        print("  docker-compose -f docker-compose.gpu.yml up -d")
        return "gpu"
    
    elif is_apple_silicon:
        print("🚀 Recommended: CPU Configuration (optimized for Apple Silicon)")
        print("\nRun the application with:")
        print("  docker-compose up -d")
        return "cpu-apple"
    
    else:
        print("🚀 Recommended: CPU Configuration")
        print("\nRun the application with:")
        print("  docker-compose up -d")
        return "cpu"

def main():
    """Main function to check environment and make recommendations."""
    print_header("Sailing Assistant Environment Checker")
    
    system, machine, is_apple_silicon = check_platform()
    docker_ok = check_docker()
    has_nvidia, has_cuda = check_gpu()
    
    if has_nvidia:
        nvidia_docker = check_nvidia_docker()
    else:
        nvidia_docker = False
    
    if not docker_ok:
        print_header("Docker Issue Detected")
        print("Please install Docker and Docker Compose before continuing.")
        print("Visit https://docs.docker.com/get-docker/ for installation instructions.")
        return
    
    if has_nvidia and not nvidia_docker:
        print_header("NVIDIA Docker Issue Detected")
        print("You have an NVIDIA GPU, but the NVIDIA Docker runtime is not available.")
        print("Please install the NVIDIA Container Toolkit:")
        print("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
    
    config = recommend_configuration(is_apple_silicon, has_nvidia, has_cuda)
    
    print("\nAdditional Notes:")
    if config == "gpu":
        print("- Make sure you have at least 8GB of GPU memory")
        print("- The application will use CUDA for faster processing")
        print("- See GPU_SETUP.md for detailed setup instructions")
    elif config == "cpu-apple":
        print("- The application will use optimized CPU code for Apple Silicon")
        print("- Memory-mapped tensors will be used for efficient memory usage")
    else:
        print("- The application will use standard CPU processing")
        print("- This may be slower for large models")
    
    print("\nMemory Requirements:")
    print("- At least 8GB of RAM is recommended")
    print("- For GPU mode, at least 8GB of GPU memory is recommended")
    
    print_header("Environment Check Complete")

if __name__ == "__main__":
    main() 