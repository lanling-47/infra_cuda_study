#!/bin/bash
# Build custom CUDA ops for Whisper optimization

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
echo "Building Custom CUDA Operations"
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

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Build extension
echo ""
echo "Building extension..."
python3 setup.py install --user

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import custom_cuda_ops; print('✓ custom_cuda_ops loaded successfully')" || {
    echo "ERROR: Failed to load custom_cuda_ops"
    exit 1
}

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
