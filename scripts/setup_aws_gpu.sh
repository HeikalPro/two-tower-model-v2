#!/bin/bash
# Setup script for AWS GPU instance
# Run this once on your AWS GPU instance after cloning the repository

echo "Setting up Two-Tower Recommendation System on AWS GPU..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Information:"
    nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements-gpu.txt

# Note about FAISS-GPU
echo ""
echo "Note: FAISS-CPU is installed by default. For GPU-accelerated FAISS:"
echo "  conda install -c conda-forge faiss-gpu"
echo ""

# Verify GPU availability
echo "Verifying GPU setup..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Setup complete! Activate the environment with: source venv/bin/activate"
echo "Verify GPU with: python -c \"import torch; print(torch.cuda.is_available())\""

