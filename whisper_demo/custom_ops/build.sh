#!/bin/bash
# Build custom CUDA ops for Whisper optimization (V2)

# Set CUDA path if not in PATH
if ! command -v nvcc &> /dev/null; then
    if [ -d "/usr/local/cuda-12.8/bin" ]; then
        export PATH="/usr/local/cuda-12.8/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
    elif [ -d "/usr/local/cuda/bin" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    fi
fi

echo "=========================================="
echo "Building Custom CUDA Operations (V2)"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA compiler: $(nvcc --version | grep release)"

# Check if PyTorch is installed
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not installed or CUDA not available"
    exit 1
}

# Install ninja for faster builds if not present
if ! command -v ninja &> /dev/null; then
    echo "Installing ninja for faster builds..."
    pip3 install --user ninja 2>/dev/null
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Set CUDA arch for RTX 2080 Ti (Turing)
export TORCH_CUDA_ARCH_LIST="7.5"

# Build extension
echo ""
echo "Building extension..."
python3 setup.py install --user

# Verify installation
echo ""
echo "Verifying installation..."
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
python3 -c "import custom_cuda_ops; print('✓ custom_cuda_ops loaded successfully'); print(f'  Available functions: {[x for x in dir(custom_cuda_ops) if not x.startswith(\"_\")]}')" || {
    echo "ERROR: Failed to load custom_cuda_ops"
    exit 1
}

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
