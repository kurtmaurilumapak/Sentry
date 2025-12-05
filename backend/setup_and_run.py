"""
Setup script that installs dependencies with progress indicators,
then starts the backend server.
"""

import subprocess
import sys
import os

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("  Sentry Backend - Dependency Installation")
    print("=" * 60)
    print()

def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is available."""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            # Try to extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    # Extract CUDA version (e.g., "12.5" from "CUDA Version: 12.5")
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        cuda_version = parts[1].strip().split()[0]
                        return True, cuda_version
            return True, None
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, None

def check_pytorch_cuda():
    """Check if PyTorch with CUDA is installed."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_pytorch_with_cuda(cuda_version=None):
    """Install PyTorch with CUDA support."""
    print("[GPU] NVIDIA GPU detected!")
    
    # Check if CPU-only PyTorch is installed and needs to be removed
    try:
        import torch
        if not torch.cuda.is_available() and "+cpu" in torch.__version__:
            print("   Detected CPU-only PyTorch - upgrading to CUDA version...")
            print("   Uninstalling CPU-only PyTorch...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                capture_output=True
            )
    except ImportError:
        pass  # PyTorch not installed yet
    
    # Determine CUDA version to use
    if cuda_version:
        # Map detected CUDA version to PyTorch CUDA version
        try:
            cuda_major = int(cuda_version.split('.')[0])
            if cuda_major >= 12:
                # Use CUDA 12.4 (compatible with CUDA 12.x)
                pytorch_cuda = "cu124"
            elif cuda_major == 11:
                # Use CUDA 11.8
                pytorch_cuda = "cu118"
            else:
                # Fallback to CUDA 12.4
                pytorch_cuda = "cu124"
        except (ValueError, IndexError):
            pytorch_cuda = "cu124"
    else:
        # Default to CUDA 12.4
        pytorch_cuda = "cu124"
    
    print(f"   Installing PyTorch with CUDA support ({pytorch_cuda})...")
    print("   This may take several minutes (downloading ~2.5GB)...")
    print("-" * 60)
    
    try:
        index_url = f"https://download.pytorch.org/whl/{pytorch_cuda}"
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", index_url
            ],
            check=True
        )
        print("-" * 60)
        print("[OK] PyTorch with CUDA installed successfully!")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print("[WARNING] Failed to install PyTorch with CUDA")
        print("   Will fall back to CPU version")
        print()
        return False

def check_dependencies():
    """Check if dependencies are already installed."""
    try:
        import fastapi
        import uvicorn
        import ultralytics
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies with progress indicator."""
    print("[*] Checking dependencies...")
    print()

    # Allow forcing CPU-only mode to avoid GPU/torch init crashes
    cpu_only_env = os.environ.get("SENTRY_CPU_ONLY") or os.environ.get("SENTRY_FORCE_CPU")
    force_cpu = str(cpu_only_env).lower() in {"1", "true", "yes", "on"}
    
    if force_cpu:
        print("[*] CPU-only mode enabled via SENTRY_CPU_ONLY/SENTRY_FORCE_CPU")
        has_nvidia_gpu, cuda_version = False, None
    else:
        # Detect NVIDIA GPU before installing
        print("[*] Detecting hardware...")
        has_nvidia_gpu, cuda_version = detect_nvidia_gpu()
    
    # Check if already installed
    deps_installed = check_dependencies()
    
    if deps_installed:
        # Check if PyTorch CUDA is available
        if not force_cpu and check_pytorch_cuda():
            print("[OK] Dependencies already installed (with CUDA support)")
            print()
            return True
        elif has_nvidia_gpu and not force_cpu:
            # Dependencies installed but GPU detected and CUDA not available
            # Upgrade to CUDA version
            print("[WARNING] GPU detected but CPU-only PyTorch installed")
            print("   Upgrading to CUDA version...")
            print()
            if install_pytorch_with_cuda(cuda_version):
                print("[OK] Upgraded to PyTorch with CUDA!")
            print()
            return True
        else:
            print("[OK] Dependencies already installed (CPU mode)")
            print()
            return True
    
    # Fresh installation - install PyTorch with CUDA if GPU detected
    if has_nvidia_gpu and not force_cpu:
        # Check if PyTorch with CUDA is already installed
        if not check_pytorch_cuda():
            # Install PyTorch with CUDA first
            install_pytorch_with_cuda(cuda_version)
        else:
            print("[OK] PyTorch with CUDA already installed")
            print()
    else:
        if force_cpu:
            print("[*] CPU-only mode enforced (skipping GPU detection/initialization)")
        else:
            print("[*] No NVIDIA GPU detected - will use CPU mode")
        print()
    
    print("[*] Installing other dependencies (this may take a few minutes)...")
    print("   Installing packages from requirements.txt...")
    print("-" * 60)
    
    try:
        # Use pip install with verbose output for better progress visibility
        # Remove -q flag to show installation progress
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
        print("-" * 60)
        print("[OK] Dependencies installed successfully!")
        
        # Verify CUDA if GPU was detected
        if has_nvidia_gpu:
            if check_pytorch_cuda():
                import torch
                print(f"[OK] GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            else:
                print("[WARNING] GPU detected but CUDA not available - using CPU mode")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print("[ERROR] Error installing dependencies!")
        print(f"   Error code: {e.returncode}")
        print("   Please check the error messages above.")
        print()
        return False
    except FileNotFoundError:
        print("-" * 60)
        print("[ERROR] Error: pip not found!")
        print("   Please ensure Python and pip are installed and in PATH.")
        print()
        return False

def main():
    """Main setup and run function."""
    print_header()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Start the backend
    print("[*] Starting backend server...")
    print("=" * 60)
    print()
    
    try:
        # Import and run main
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n\nBackend stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error starting backend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
