# Two-Tower Recommendation Retrieval System

A production-grade recommendation engine for e-commerce platforms with Arabic text support, implementing a Two-Tower architecture for efficient product retrieval.

## Architecture Overview

The system uses a **Two-Tower** model architecture:

- **Item Tower**: Content-based encoder that processes Arabic product text (title + description) using multilingual sentence-transformers
- **Buyer Tower**: Behavior-based encoder that aggregates buyer interaction sequences with weighted event importance

Both towers output L2-normalized embeddings in the same space, enabling efficient cosine similarity-based retrieval.

## Features

- ✅ Arabic-first text encoding with multilingual sentence-transformers
- ✅ Configurable aggregation methods (weighted average or attention)
- ✅ Event-weighted buyer behavior modeling (view=1, add_to_cart=5, purchase=10)
- ✅ FAISS-based vector database for fast retrieval
- ✅ Real-time API endpoint for buyer encoding and product retrieval
- ✅ Offline batch processing for product embeddings
- ✅ Production-ready with validation and sanity checks

## Project Structure

```
Two Tower model v2/
├── src/
│   ├── data/           # Data processing and dataset
│   ├── models/         # Model architectures
│   ├── training/      # Training pipeline and losses
│   ├── inference/     # Embedding generation and vector DB
│   ├── api/           # FastAPI server
│   └── utils/         # Configuration management
├── configs/           # Configuration files
├── scripts/            # Training and utility scripts
├── tests/             # Sanity checks and tests
└── data/              # Input data (events.csv, products.csv)
```

## Installation

### Local Setup (CPU)

1. **Clone the repository** (or ensure you're in the project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; import sentence_transformers; import faiss; print('All dependencies installed!')"
```

### AWS GPU Setup

1. **SSH into your AWS GPU instance**

2. **Clone the repository**:
```bash
git clone <your-repo-url>
cd "Two Tower model v2"
```

3. **Run the GPU setup script**:
```bash
chmod +x scripts/setup_aws_gpu.sh
./scripts/setup_aws_gpu.sh
```

Or manually:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust version based on your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements-gpu.txt
```

4. **Verify GPU availability**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

5. **Update config for GPU**:
Ensure `configs/config.yaml` has:
```yaml
inference:
  device: "cuda"  # Use GPU
```

## Configuration

Edit `configs/config.yaml` to customize:

- Model hyperparameters (embedding dimensions, aggregation method)
- Training settings (batch size, learning rate, epochs)
- Event weights (view, add_to_cart, purchase)
- Data paths and output directories

## Usage

### 1. Training

Train the Two-Tower model on your data:

```bash
python scripts/train.py
```

This will:
- Load and process events and products data
- Create training/validation splits
- Train the model with InfoNCE loss
- Save checkpoints to `checkpoints/`

### 2. Generate Product Embeddings

After training, generate embeddings for all products:

```bash
python scripts/generate_embeddings.py
```

This creates:
- `outputs/embeddings/product_embeddings.npy`
- `outputs/embeddings/product_ids.npy`
- `outputs/embeddings/product_id_to_index.json`

### 3. Build FAISS Index

Create the vector database index:

```bash
python scripts/build_index.py
```

This creates:
- `outputs/index/product_index.faiss`
- `outputs/index/product_ids.npy`
- `outputs/index/product_id_to_index.json`

### 4. Start API Server

Launch the FastAPI server:

```bash
python -m src.api.server
```

Or using uvicorn directly:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 5. API Usage

#### Encode Buyer

```bash
curl -X POST "http://localhost:8000/encode_buyer" \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_id": "buyer_123",
    "recent_interactions": [
      {"product_id": "prod_1", "event_type": "view"},
      {"product_id": "prod_1", "event_type": "add_to_cart"},
      {"product_id": "prod_2", "event_type": "purchase"}
    ]
  }'
```

#### Retrieve Products

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_id": "buyer_123",
    "recent_interactions": [
      {"product_id": "prod_1", "event_type": "view"},
      {"product_id": "prod_1", "event_type": "purchase"}
    ],
    "k": 10
  }'
```

## Sanity Checks

Run validation tests to verify the system:

```bash
python tests/test_sanity_checks.py
```

Tests verify:
1. **Item Embedding Similarity**: Similar items (e.g., "خاتم ذهب" vs "سلسال ذهب") have high similarity, while dissimilar items have low similarity
2. **Buyer Behavior Retrieval**: Buyers with repeated interactions retrieve relevant products

## Data Format

### Events CSV

Required columns:
- `distinct_id` or `buyer_id`: Buyer identifier
- `product_id`: Product identifier
- `event_name` or `event_type`: Event type (View, AddToCart, Purchase)
- `created_at` or `timestamp`: Event timestamp

### Products CSV

Required columns:
- `id` or `product_id`: Product identifier
- `title`: Product title (Arabic)
- `description`: Product description (Arabic)
- `metadata`: JSON string with optional `brand` and `catalog_id` (category)

## Model Architecture Details

### Item Tower

- **Input**: Product title + description (Arabic text)
- **Encoder**: Multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Optional**: Categorical embeddings for brand and category
- **Output**: 384-dimensional L2-normalized embedding

### Buyer Tower

- **Input**: Sequence of interacted product IDs with event weights
- **Aggregation**: Configurable (weighted average or attention mechanism)
- **Event Weights**: view=1, add_to_cart=5, purchase=10
- **Output**: 384-dimensional L2-normalized embedding

### Training

- **Loss**: InfoNCE (contrastive loss) with temperature scaling
- **Negative Sampling**: In-batch negatives + random negatives
- **Optimization**: Adam optimizer

## Performance Considerations

- **GPU Acceleration**: All tensors are automatically moved to GPU when available. Training is significantly faster on GPU (10-50x speedup).
- **Text Encoder**: Pretrained model is frozen by default (faster training)
- **Batch Processing**: Product embeddings generated offline for efficiency
- **FAISS Index**: Fast approximate nearest neighbor search (use `faiss-gpu` for GPU acceleration)
- **API Latency**: Target <100ms for buyer encoding + retrieval

## Future Enhancements

- **Phase 2**: Re-ranking stage for final product ordering
- **A/B Testing**: Support for multiple model versions
- **Monitoring**: Metrics tracking and logging
- **Caching**: Buyer embedding caching for frequently accessed users

## Troubleshooting

### Model checkpoint not found

Ensure you've trained the model first:
```bash
python scripts/train.py
```

### FAISS index not found

Generate embeddings and build index:
```bash
python scripts/generate_embeddings.py
python scripts/build_index.py
```

### Out of memory errors

Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 256  # Reduce from 512
```

## License

This project is designed for production use in e-commerce recommendation systems.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

