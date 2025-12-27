"""Script to create .env file from template."""

import shutil
from pathlib import Path


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    if env_file.exists():
        print(".env file already exists. Skipping creation.")
        return
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"Created .env file from .env.example at {env_file}")
        print("Please edit .env file and fill in your actual values.")
    else:
        print("Error: .env.example not found. Creating basic .env file...")
        # Create basic .env file
        env_content = """# Environment Variables for Two-Tower Recommendation System

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
AWS_INSTANCE_IP=your_aws_instance_ip_here
AWS_SSH_USER=ubuntu

# Data Paths (if using cloud storage)
DATA_BUCKET=s3://your-bucket-name/data
EMBEDDINGS_BUCKET=s3://your-bucket-name/embeddings

# Model Configuration
MODEL_CHECKPOINT_PATH=checkpoints/best_model.pt
EMBEDDINGS_DIR=outputs/embeddings
INDEX_DIR=outputs/index

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Training Configuration
CUDA_VISIBLE_DEVICES=0
NUM_WORKERS=4
PIN_MEMORY=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Optional: Hugging Face Token (if using private models)
HF_TOKEN=your_huggingface_token_here
"""
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"Created basic .env file at {env_file}")
        print("Please edit .env file and fill in your actual values.")


if __name__ == "__main__":
    create_env_file()

