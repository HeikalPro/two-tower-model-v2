---
name: Two-Tower Recommendation Retrieval System
overview: Build a production-grade Two-Tower retrieval model for e-commerce recommendations with Arabic text support, including training pipeline, embedding generation, vector database indexing, and real-time API endpoint.
todos:
  - id: setup_project
    content: Create project structure with directories (src/, configs/, scripts/, tests/) and __init__.py files
    status: completed
  - id: data_processor
    content: Implement data processor to parse events.csv and products.csv, extract metadata, create interaction sequences with event weights
    status: completed
    dependencies:
      - setup_project
  - id: item_tower
    content: Implement Item Tower with multilingual sentence-transformer, Arabic text encoding, optional categorical features, and L2 normalization
    status: completed
    dependencies:
      - setup_project
  - id: buyer_tower
    content: Implement Buyer Tower with configurable weighted average and attention aggregation methods, event weighting, and L2 normalization
    status: completed
    dependencies:
      - setup_project
      - item_tower
  - id: two_tower_model
    content: Combine Item and Buyer towers into unified Two-Tower model with shared item embeddings
    status: completed
    dependencies:
      - item_tower
      - buyer_tower
  - id: training_pipeline
    content: Implement training pipeline with InfoNCE loss, negative sampling, checkpointing, and validation
    status: completed
    dependencies:
      - two_tower_model
      - data_processor
  - id: embedding_generation
    content: Implement offline item embedding generation and online buyer embedding inference
    status: completed
    dependencies:
      - two_tower_model
  - id: vector_database
    content: Implement FAISS index creation, product embedding storage, and Top-K retrieval functionality
    status: completed
    dependencies:
      - embedding_generation
  - id: api_endpoint
    content: Implement FastAPI server with /encode_buyer and /retrieve endpoints for real-time inference
    status: completed
    dependencies:
      - embedding_generation
      - vector_database
  - id: sanity_checks
    content: Implement validation tests for item embedding similarity and buyer behavior retrieval accuracy
    status: completed
    dependencies:
      - embedding_generation
      - vector_database
  - id: config_management
    content: Create configuration YAML file with all hyperparameters, paths, and settings
    status: completed
    dependencies:
      - setup_project
  - id: training_scripts
    content: Create training script, embedding generation script, and index building script
    status: completed
    dependencies:
      - training_pipeline
      - embedding_generation
      - vector_database
      - config_management
  - id: requirements_docs
    content: Create requirements.txt with all dependencies and README.md with setup and usage instructions
    status: completed
    dependencies:
      - setup_project
---

# Two-Tower Recommendation Retrieval System

## Architecture Overview

```mermaid
graph TB
    subgraph "Data Layer"
        Events[events.csv<br/>buyer interactions]
        Products[products.csv<br/>Arabic product metadata]
    end
    
    subgraph "Training Pipeline"
        DataProc[Data Processor<br/>Parse & Clean]
        ItemTower[Item Tower<br/>Arabic Text Encoder]
        BuyerTower[Buyer Tower<br/>Weighted Aggregation]
        Trainer[Contrastive Training<br/>InfoNCE Loss]
    end
    
    subgraph "Inference Pipeline"
        ItemEmb[Item Embeddings<br/>Precomputed]
        FAISS[FAISS Vector Index]
        BuyerAPI[Buyer Encoding API<br/>Real-time]
        Retrieval[Top-K Retrieval]
    end
    
    Events --> DataProc
    Products --> DataProc
    DataProc --> ItemTower
    DataProc --> BuyerTower
    ItemTower --> Trainer
    BuyerTower --> Trainer
    Trainer --> ItemEmb
    ItemEmb --> FAISS
    BuyerAPI --> Retrieval
    FAISS --> Retrieval
```



## Implementation Plan

### 1. Project Structure

```javascript
Two Tower model v2/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── processor.py          # Data loading and preprocessing
│   │   └── dataset.py            # PyTorch dataset for training
│   ├── models/
│   │   ├── __init__.py
│   │   ├── item_tower.py         # Item Tower (content-based)
│   │   ├── buyer_tower.py        # Buyer Tower (behavior-based)
│   │   └── two_tower.py          # Combined model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   └── losses.py             # Contrastive loss functions
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── encoder.py            # Embedding generation
│   │   └── vector_db.py          # FAISS index management
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py             # FastAPI endpoint
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Configuration management
├── configs/
│   └── config.yaml               # Hyperparameters and settings
├── scripts/
│   ├── train.py                  # Training script
│   ├── generate_embeddings.py    # Batch embedding generation
│   └── build_index.py            # FAISS index creation
├── tests/
│   ├── test_item_tower.py
│   ├── test_buyer_tower.py
│   └── test_sanity_checks.py
├── requirements.txt
└── README.md
```



### 2. Data Processing ([src/data/processor.py](src/data/processor.py))

- **Event Processing**:
- Parse `events.csv` with columns: `distinct_id` (buyer_id), `product_id`, `event_name`, `created_at`
- Map event names: `View`→1, `AddToCart`→5, `Purchase`→10
- Sort by timestamp, create interaction sequences per buyer
- Handle missing/invalid data
- **Product Processing**:
- Parse `products.csv` with Arabic `title` and `description`
- Extract `brand` and `category` from JSON `metadata` field
- Combine `title + description` for text input
- Create product ID to metadata mapping
- **Dataset Creation**:
- Positive pairs: (buyer_id, product_id) from interactions
- Negative sampling: in-batch negatives + random product negatives
- Batch generation for training

### 3. Item Tower ([src/models/item_tower.py](src/models/item_tower.py))

- **Text Encoder**:
- Use `sentence-transformers` with multilingual model: `paraphrase-multilingual-MiniLM-L12-v2`
- Input: concatenated `title + " " + description` (Arabic text)
- Output: 384-dimensional embedding (model default)
- **Optional Categorical Features**:
- Embedding layers for `brand` and `category` (if provided)
- Concatenate with text embedding
- Projection layer to final embedding dimension
- **Normalization**:
- L2-normalize final item embeddings
- Output dimension: 384 (or configurable)

### 4. Buyer Tower ([src/models/buyer_tower.py](src/models/buyer_tower.py))

- **Input Processing**:
- Receive sequence of `(product_id, event_weight)` tuples
- Map product IDs to item embeddings (from Item Tower)
- **Aggregation Methods** (configurable):
- **Weighted Average**: Simple weighted sum with L2 normalization
- **Attention Mechanism**: Learnable attention weights over interaction sequence
    - Query: learnable buyer context vector
    - Keys/Values: weighted item embeddings
    - Output: attended aggregation
- **Normalization**:
- L2-normalize final buyer embedding
- Same dimension as item embeddings

### 5. Training Pipeline ([src/training/trainer.py](src/training/trainer.py))

- **Loss Function**:
- InfoNCE (contrastive loss) with temperature scaling
- Positive: cosine similarity between buyer and interacted product
- Negatives: in-batch negatives + sampled random negatives
- **Training Loop**:
- Freeze Item Tower text encoder (pretrained), fine-tune projection layers
- Train Buyer Tower from scratch
- Batch size, learning rate, epochs configurable
- Validation split for monitoring
- **Checkpointing**:
- Save model checkpoints
- Track best model based on validation metrics

### 6. Embedding Generation ([src/inference/encoder.py](src/inference/encoder.py))

- **Item Embeddings (Offline)**:
- Load trained Item Tower
- Process all products in batch
- Generate and save embeddings for all products
- Output: `product_embeddings.npy` + `product_ids.npy`
- **Buyer Embeddings (Online)**:
- Load trained Buyer Tower
- Accept recent interactions (last N events)
- Generate buyer embedding in real-time
- Stateless inference

### 7. Vector Database ([src/inference/vector_db.py](src/inference/vector_db.py))

- **FAISS Index**:
- Build FAISS IndexFlatIP (inner product for cosine similarity)
- Add all product embeddings (L2-normalized)
- Save index to disk: `product_index.faiss`
- Load index for retrieval
- **Retrieval**:
- Given buyer embedding, retrieve Top-K products
- Return product IDs and similarity scores
- Efficient batch retrieval support

### 8. API Endpoint ([src/api/server.py](src/api/server.py))

- **FastAPI Server**:
- `POST /encode_buyer`:
    - Input: `{"buyer_id": str, "recent_interactions": [{"product_id": str, "event_type": str, "timestamp": str}]}`
    - Output: `{"buyer_embedding": [float], "dimension": int}`
- `GET /retrieve`:
    - Input: `{"buyer_id": str, "k": int}`
    - Output: `{"product_ids": [str], "scores": [float]}`
- Health check endpoint
- **Performance**:
- Model loading at startup
- Fast inference (<100ms target)
- Async support

### 9. Sanity Checks ([tests/test_sanity_checks.py](tests/test_sanity_checks.py))

- **Item Embedding Validation**:
- Test: `similarity("خاتم ذهب", "سلسال ذهب")` → HIGH (>0.7)
- Test: `similarity("خاتم ذهب", "زيت محرك")` → LOW (<0.3)
- **Buyer Behavior Validation**:
- Simulate buyer with repeated gold ring interactions
- Verify Top-K retrieval contains jewelry items
- Verify no unrelated categories (cars, food) in top results

### 10. Configuration ([configs/config.yaml](configs/config.yaml))

- Model hyperparameters (embedding dim, hidden dims)
- Training settings (batch size, learning rate, epochs)
- Event weights (view, add_to_cart, purchase)
- Aggregation method (weighted_avg vs attention)
- Paths for data, checkpoints, embeddings

## Key Design Decisions

1. **Multilingual Sentence-Transformers**: Balance between Arabic support and ease of deployment
2. **Dual Aggregation Methods**: Configurable weighted average (fast) and attention (expressive)
3. **Separate Training/Inference**: Clear separation for production deployment
4. **FAISS for Retrieval**: Industry-standard, efficient vector search
5. **Stateless API**: Real-time buyer encoding without retraining