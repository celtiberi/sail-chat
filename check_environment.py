#!/usr/bin/env python3
"""
Environment checker script to determine the appropriate Docker configuration.
This script checks the system's hardware and software to recommend the best
Docker configuration for running the sailing assistant application.
"""

import platform
import os
import subprocess
import sys

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_section(text):
    """Print a section header."""
    print("\n" + "-" * 40)
    print(f" {text}")
    print("-" * 40)

def check_platform():
    """Check the system platform and architecture."""
    system = platform.system()
    machine = platform.machine()
    
    print_section("System Information")
    print(f"Operating System: {system}")
    print(f"Architecture: {machine}")
    
    is_apple_silicon = system == "Darwin" and machine.startswith("arm")
    if is_apple_silicon:
        print("✅ Detected Apple Silicon Mac")
    elif system == "Darwin":
        print("✅ Detected Intel Mac")
    elif system == "Linux":
        print("✅ Detected Linux")
    elif system == "Windows":
        print("✅ Detected Windows")
    
    return system, machine, is_apple_silicon

def check_docker():
    """Check if Docker is installed and running."""
    print_section("Docker Check")
    
    try:
        docker_version = subprocess.check_output(["docker", "--version"], 
                                               stderr=subprocess.STDOUT,
                                               universal_newlines=True).strip()
        print(f"✅ Docker installed: {docker_version}")
        
        # Check if Docker is running
        subprocess.check_output(["docker", "info"], 
                              stderr=subprocess.STDOUT,
                              universal_newlines=True)
        print("✅ Docker daemon is running")
        
        # Check Docker Compose
        try:
            compose_version = subprocess.check_output(["docker", "compose", "version"], 
                                                   stderr=subprocess.STDOUT,
                                                   universal_newlines=True).strip()
            print(f"✅ Docker Compose installed: {compose_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try legacy docker-compose command
                compose_version = subprocess.check_output(["docker-compose", "--version"], 
                                                       stderr=subprocess.STDOUT,
                                                       universal_newlines=True).strip()
                print(f"✅ Docker Compose installed (legacy): {compose_version}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Docker Compose not found")
                return False
        
        return True
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not installed or not running")
        return False

def check_gpu():
    """Check for NVIDIA GPU and CUDA support."""
    print_section("GPU Check")
    
    has_nvidia = False
    has_cuda = False
    
    # Check for NVIDIA GPU on Linux
    if platform.system() == "Linux":
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi"], 
                                               stderr=subprocess.STDOUT,
                                               universal_newlines=True)
            print("✅ NVIDIA GPU detected")
            print(nvidia_smi.split("\n")[2:4])
            has_nvidia = True
            
            # Check for nvidia-docker
            try:
                subprocess.check_output(["docker", "info"], 
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
                if "nvidia" in subprocess.check_output(["docker", "info"], 
                                                     stderr=subprocess.STDOUT,
                                                     universal_newlines=True):
                    print("✅ NVIDIA Docker runtime detected")
                else:
                    print("❌ NVIDIA Docker runtime not detected")
            except subprocess.CalledProcessError:
                print("❌ Could not check for NVIDIA Docker runtime")
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ No NVIDIA GPU detected or nvidia-smi not installed")
    
    # Check for CUDA via PyTorch if available
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA not available in PyTorch")
            
        # Check for MPS (Apple Silicon)
        if platform.system() == "Darwin" and platform.machine().startswith("arm"):
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
                if torch.mps.is_available():
                    print("✅ MPS (Metal Performance Shaders) available")
                else:
                    print("❌ MPS not available")
            else:
                print("❌ PyTorch version does not support MPS")
    
    except ImportError:
        print("❓ PyTorch not installed, cannot check CUDA availability")
    
    return has_nvidia, has_cuda

def recommend_configuration(is_apple_silicon, has_nvidia, has_cuda):
    """Recommend the appropriate Docker configuration."""
    print_header("Recommendation")
    
    if has_nvidia and has_cuda:
        print("✅ Recommended configuration: GPU mode")
        print("\nRun the application with:")
        print("  docker-compose -f docker-compose.gpu.yml up -d")
        return "gpu"
    elif is_apple_silicon:
        print("✅ Recommended configuration: CPU mode (optimized for Apple Silicon)")
        print("\nRun the application with:")
        print("  docker-compose up -d")
        return "cpu-apple"
    else:
        print("✅ Recommended configuration: CPU mode")
        print("\nRun the application with:")
        print("  docker-compose up -d")
        return "cpu"

def main():
    """Main function to check environment and make recommendations."""
    print_header("Sailing Assistant Environment Checker")
    
    system, machine, is_apple_silicon = check_platform()
    docker_ok = check_docker()
    has_nvidia, has_cuda = check_gpu()
    
    if not docker_ok:
        print_header("Docker Issue Detected")
        print("Please install Docker and Docker Compose before continuing.")
        print("Visit https://docs.docker.com/get-docker/ for installation instructions.")
        return
    
    config = recommend_configuration(is_apple_silicon, has_nvidia, has_cuda)
    
    print("\nAdditional Notes:")
    if config == "gpu":
        print("- Make sure you have at least 8GB of GPU memory")
        print("- The application will use CUDA for faster processing")
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