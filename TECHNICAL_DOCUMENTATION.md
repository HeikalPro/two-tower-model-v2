# Two-Tower Recommendation System - Technical Documentation

## 1. Project Overview

### Problem Statement
E-commerce recommendation system requiring real-time product retrieval from large catalogs (100K+ items) with Arabic text support. The system must encode buyer behavioral patterns and product content into a shared embedding space for efficient similarity search.

### Core Objective
Two-tower neural architecture: Item Tower encodes product text/metadata, Buyer Tower encodes interaction sequences. Both output 384-dimensional L2-normalized embeddings. Retrieval via cosine similarity using FAISS vector database.

### Design Philosophy
- **Offline embedding generation**: Product embeddings pre-computed; only buyer embeddings computed at inference time
- **Configurable aggregation**: Buyer Tower supports weighted average or attention-based aggregation
- **Event weighting**: Purchase (10) > AddToCart (5) > View (1)
- **Text encoder frozen**: Pre-trained multilingual sentence transformer frozen by default for faster training
- **Batch optimization**: Training batches buyer sequence encodings to avoid per-sequence forward passes

---

## 2. System Architecture

### Component-Level Architecture

```
┌─────────────────────────────────────────────┐
│         Data Layer (CSV Files)              │
│  events.csv | products.csv | photos.csv     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      DataProcessor (src/data/processor.py)  │
│  • Column normalization                      │
│  • Metadata extraction (JSON parsing)       │
│  • Text concatenation (title + description) │
│  • Deduplication (content-based)            │
│  • Sequence creation (time-ordered)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    Training Pipeline (scripts/train.py)     │
│  • TwoTowerDataset (positive + negatives)   │
│  • Trainer (InfoNCE loss, Adam optimizer)   │
│  • Checkpointing (epoch-based + best model) │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│   Embedding Generation (scripts/...)        │
│  • EmbeddingEncoder.encode_items()          │
│  • Batch processing (64 products/batch)     │
│  • FAISS index construction                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      API Server (src/api/server.py)         │
│  • FastAPI endpoints                         │
│  • Buyer encoding (real-time)               │
│  • Product retrieval (FAISS search)         │
│  • Product metadata lookup (DataFrame)      │
└─────────────────────────────────────────────┘
```

### Module Responsibilities

**Item Tower** (`src/models/item_tower.py`):
- Encodes product text using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim output)
- Optional categorical embeddings (brand, category) - 64-dim each
- Projection layer: text_dim + 2×64 → 256 → 384 with ReLU + Dropout(0.1)
- Output: L2-normalized 384-dim vector

**Buyer Tower** (`src/models/buyer_tower.py`):
- Aggregates sequence of item embeddings (from Item Tower)
- Two modes: `weighted_avg` (event weights normalized) or `attention` (learned attention + event weights)
- Attention: 384 → 128 → 1, softmax over sequence, multiplied by event weights
- Output: L2-normalized 384-dim vector

**Training** (`src/training/trainer.py`):
- InfoNCE loss: positive similarity + in-batch negatives + random negatives
- Batch optimization: encodes all buyer sequence texts in single forward pass
- Device management: automatic GPU/CPU based on `config['inference']['device']`
- Checkpoint format: model state, optimizer state, epoch, vocabularies (brand/category)

**Inference** (`src/inference/encoder.py`):
- Model loading: reconstructs ItemTower/BuyerTower from checkpoint, handles categorical vocab reconstruction
- Product metadata required: must call `set_product_metadata()` before encoding
- Buyer encoding: limits to `max_interaction_history` (100), sorts by timestamp if available

**Vector Database** (`src/inference/vector_db.py`):
- FAISS IndexFlatIP: inner product for normalized vectors = cosine similarity
- Embeddings re-normalized on build (safety check)
- Product ID mapping: JSON file stores id→index mapping

### Runtime Flow

**Training:**
1. `DataProcessor.load_events()`: chunked CSV reading, column normalization, timestamp parsing
2. `DataProcessor.create_interaction_sequences()`: groups by buyer_id, applies max_history truncation
3. `TwoTowerDataset`: samples random negatives per positive pair
4. `Trainer._encode_buyer_sequences_batched()`: collects all sequence texts, single batch encoding
5. `TwoTowerModel.forward_simplified()`: buyer/item embeddings → InfoNCE loss

**Inference (API):**
1. Startup: loads model checkpoint, FAISS index, product DataFrame, photo mapping
2. Request `/retrieve`: interactions → `encoder.encode_buyer()` → FAISS search → DataFrame lookup
3. Buyer encoding: interactions sorted by timestamp, limited to max_history, item embeddings aggregated
4. Retrieval: query embedding L2-normalized, FAISS returns top-K with scores

---

## 3. Directory & Module Breakdown

### Top-Level Structure

```
Two Tower model v2/
├── src/                    # Source code (Python package)
├── scripts/                # Executable scripts (entry points)
├── configs/                # YAML configuration files
├── data/                   # Input data (CSV files)
├── outputs/                # Generated artifacts (embeddings, index, eval)
├── checkpoints/            # Model checkpoints (.pt files)
├── tests/                  # Test files (sanity checks)
└── requirements*.txt       # Python dependencies
```

### Source Modules (`src/`)

**`src/data/`**:
- `processor.py`: Data loading, cleaning, sequence creation, deduplication
- `dataset.py`: PyTorch Dataset, negative sampling, collate function

**`src/models/`**:
- `item_tower.py`: Sentence transformer wrapper, categorical embeddings, projection
- `buyer_tower.py`: Aggregation logic (weighted_avg/attention)
- `two_tower.py`: Combined model, forward methods for training/inference

**`src/training/`**:
- `trainer.py`: Training loop, validation, checkpointing, batched sequence encoding
- `losses.py`: InfoNCE loss implementation (temperature scaling, in-batch negatives)

**`src/inference/`**:
- `encoder.py`: Model loading, checkpoint reconstruction, embedding generation
- `vector_db.py`: FAISS index management, search operations

**`src/api/`**:
- `server.py`: FastAPI application, startup initialization, endpoint handlers

**`src/evaluation/`**:
- `metrics.py`: Retrieval metrics (recall, precision, NDCG, MRR), diversity, coverage, relevance

**`src/utils/`**:
- `config.py`: YAML config loading, event weight mapping
- `env_loader.py`: .env file parsing (auto-loaded on import)

### Scripts (`scripts/`)

- `train.py`: Training entry point, orchestrates data loading, model init, training
- `generate_embeddings.py`: Batch product embedding generation, saves .npy files
- `build_index.py`: Loads embeddings, builds FAISS index, saves index files
- `evaluate.py`: Evaluation runner, test data preparation, metrics computation
- `create_env_file.py`: .env file generation utility
- `setup_aws_gpu.sh`: GPU environment setup script

### Critical Files

**`configs/config.yaml`**: Single source of truth for all hyperparameters, paths, device settings. Must exist.

**`src/training/trainer.py`**: Contains batched encoding optimization (critical for training performance).

**`src/inference/encoder.py`**: Checkpoint loading logic includes categorical vocab reconstruction (handles missing vocabs from old checkpoints).

**`src/api/server.py`**: Startup event loads all artifacts; failures raise exceptions (prevents serving with missing dependencies).

---

## 4. Core Data Flow

### Training Data Flow

```
CSV Files
  → DataProcessor.load_events()
    → DataFrame (buyer_id, product_id, event_type, timestamp)
  → DataProcessor.create_interaction_sequences()
    → Dict[buyer_id, List[(product_id, weight, timestamp)]]
  → DataProcessor.create_positive_pairs()
    → List[(buyer_id, product_id, weight)]
  → TwoTowerDataset.__getitem__()
    → Samples negatives, returns buyer_sequence + texts
  → DataLoader (batch collation)
  → Trainer.train_epoch()
    → _encode_buyer_sequences_batched() [OPTIMIZATION: single batch]
      → ItemTower.encode_text() [all sequence texts in one call]
    → TwoTowerModel.forward_simplified()
      → ItemTower (positive/negative texts)
      → BuyerTower (aggregated sequence embeddings)
    → InfoNCELoss()
      → Cross-entropy over (positive + negatives + in-batch)
```

### Inference Data Flow

```
API Request (POST /retrieve)
  → server.py:retrieve()
    → interactions: List[Dict]
  → EmbeddingEncoder.encode_buyer()
    → Sorts by timestamp, limits to max_history
    → ItemTower (product texts from interactions)
      → List[embeddings] [seq_len, 384]
    → BuyerTower (aggregated)
      → [384]
  → VectorDatabase.retrieve()
    → L2-normalize query
    → FAISS index.search()
      → Returns (indices, scores)
    → Map indices → product_ids
  → DataFrame lookup (products_df)
    → Extract title, description, brand, category
  → product_photos dict lookup
  → Response (ProductInfo list)
```

### Data Transformations

**Text Processing**:
- `DataProcessor._combine_text()`: title + " " + description, handles empty fields
- Empty text fallback: single space `" "` (sentence transformer requirement)

**Deduplication**:
- `DataProcessor._deduplicate_products()`: creates normalized key (lowercase, stripped, title+description+brand)
- Keeps first occurrence, drops duplicates before embedding generation

**Sequence Creation**:
- Events sorted by timestamp
- Grouped by buyer_id
- Truncated to `max_interaction_history` (most recent)
- Event weights applied via `get_event_weight()` (handles name variations: "View"/"view"/"VIEW")

**Metadata Extraction**:
- `metadata` column parsed as JSON string
- `brand` from `metadata['brand']`, `category` from `metadata['catalog_id']`
- Missing metadata → None (categorical embeddings use index 0 = `<UNK>`)

---

## 5. Key Design Decisions

### Architecture Decisions

**Two-Tower Separation**: Item and Buyer towers trained jointly but encode independently at inference. Enables offline product embedding generation, reducing API latency.

**Frozen Text Encoder**: `freeze_text_encoder=True` by default. Pre-trained multilingual model provides strong Arabic encoding; freezing reduces trainable parameters and training time.

**Categorical Embeddings**: Optional feature (64-dim each for brand/category). Vocabularies built at runtime from product metadata. `<UNK>` token at index 0 for missing values.

**Aggregation Methods**: Buyer Tower supports `weighted_avg` (no learnable params) and `attention` (learned). Attention combines learned scores with event weights before softmax.

**L2 Normalization**: All embeddings L2-normalized. Enables cosine similarity via inner product. FAISS IndexFlatIP expects normalized vectors.

### Implementation Decisions

**Batch Optimization in Training**: `Trainer._encode_buyer_sequences_batched()` collects all sequence texts, encodes in single batch. Avoids N forward passes (N = batch size) for sequence encoding.

**Checkpoint Vocab Handling**: Checkpoints store brand/category vocab dictionaries. `EmbeddingEncoder._load_model()` reconstructs vocabs; falls back to dummy vocabs if missing (for backward compatibility), then reconstructs from product metadata.

**DataLoader num_workers=0**: Hardcoded in training scripts for Windows compatibility. Can be increased on Linux for faster data loading.

**FAISS IndexFlatIP**: Exact nearest neighbor search. No approximation (IndexIVF, etc.) used. Trade-off: O(n) search time but exact results.

**Product Photo Loading**: Optional CSV file (`data/products photos.csv`). Handles column name variations (`id`/`product_id`, `thumbnail`/`photo_link`). Missing file is non-fatal (warning only).

**Global State in API Server**: `encoder`, `vector_db`, `products_df`, `product_photos` stored as module-level globals, initialized at startup. FastAPI startup event ensures initialization before serving.

---

## 6. Configuration & Environment

### Configuration File (`configs/config.yaml`)

Single YAML file contains all configuration. Structure:

```yaml
model:
  embedding_dim: 384
  item_tower:
    text_encoder: "paraphrase-multilingual-MiniLM-L12-v2"
    use_categorical_features: true
    categorical_embedding_dim: 64
    projection_hidden_dim: 256
  buyer_tower:
    aggregation_method: "attention"  # or "weighted_avg"
    attention_hidden_dim: 128
    max_interaction_history: 100

training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 3
  temperature: 0.07  # InfoNCE temperature
  num_negatives: 4
  validation_split: 0.1
  checkpoint_dir: "checkpoints"
  save_every_n_epochs: 2
  freeze_text_encoder: true

event_weights:
  view: 1
  add_to_cart: 5
  purchase: 10

data:
  events_path: "data/events.csv"
  products_path: "data/products.csv"
  output_dir: "outputs"

inference:
  embeddings_dir: "outputs/embeddings"
  index_dir: "outputs/index"
  model_checkpoint: "checkpoints/best_model.pt"
  device: "cuda"  # or "cpu"

api:
  host: "0.0.0.0"
  port: 8000
  max_interactions_per_request: 100
```

**Config Loading**: `src/utils/config.py` uses `yaml.safe_load()`. Raises `FileNotFoundError` if config missing. No environment variable substitution; paths relative to project root.

### Environment Variables (Optional)

`.env` file supported via `src/utils/env_loader.py`. Auto-loaded on import (called in `__init__`). Currently not used by core code; provided for future AWS/deployment integration.

**Environment Variable Access**: `env_loader.get_env(key, default)` loads `.env` and returns value. File location: project root (three levels up from `src/utils/env_loader.py`).

### Runtime Assumptions

- Working directory: project root (scripts change directory with `os.chdir(project_root)`)
- Data files: CSV files in `data/` directory with expected column names (normalized internally)
- Model checkpoint: `checkpoints/best_model.pt` must exist for inference
- FAISS index: `outputs/index/product_index.faiss` must exist for API server
- Product metadata: Must be set via `encoder.set_product_metadata()` before encoding

### External Services

None required. All dependencies are libraries (PyTorch, FAISS, FastAPI, sentence-transformers). No database, message queue, or external API calls.

---

## 7. Dependencies & Integrations

### External Libraries

**Core ML**:
- `torch>=2.0.0`: Neural network framework, tensor operations
- `sentence-transformers>=2.2.0`: Multilingual text encoder (Hugging Face transformers wrapper)
- `faiss-cpu>=1.7.4`: Vector similarity search (CPU version; `faiss-gpu` available via conda)

**Data Processing**:
- `pandas>=2.0.0`: CSV loading, DataFrame operations
- `numpy>=1.24.0`: Array operations, embedding storage
- `scikit-learn>=1.3.0`: `train_test_split` for data splitting

**API**:
- `fastapi>=0.104.0`: Web framework, request validation
- `uvicorn[standard]>=0.24.0`: ASGI server
- `pydantic>=2.0.0`: Request/response model validation

**Utilities**:
- `pyyaml>=6.0`: Configuration file parsing
- `tqdm>=4.66.0`: Progress bars

### Internal Dependencies

**Module Dependencies** (import graph):
- `src/utils/config.py`: Used by all modules (no dependencies)
- `src/data/processor.py`: Depends on `config.py`
- `src/data/dataset.py`: Depends on `processor.py` (for metadata access)
- `src/models/item_tower.py`: Standalone (sentence-transformers only)
- `src/models/buyer_tower.py`: Standalone (PyTorch only)
- `src/models/two_tower.py`: Depends on `item_tower.py`, `buyer_tower.py`
- `src/training/losses.py`: Standalone (PyTorch only)
- `src/training/trainer.py`: Depends on `two_tower.py`, `losses.py`, `dataset.py`, `config.py`
- `src/inference/encoder.py`: Depends on `two_tower.py`, `config.py`
- `src/inference/vector_db.py`: Standalone (FAISS only)
- `src/api/server.py`: Depends on `encoder.py`, `vector_db.py`, `processor.py`, `config.py`
- `src/evaluation/metrics.py`: Depends on `encoder.py`, `vector_db.py`, `processor.py`, `config.py`

**Circular Dependencies**: None. Dependency graph is acyclic.

### Integration Points

**Training → Inference**: Checkpoint file format is the interface. Contains `model_state_dict`, `brand_vocab`, `category_vocab`, `config`.

**Embedding Generation → Index Building**: Interface is `.npy` files (`product_embeddings.npy`, `product_ids.npy`) and JSON mapping (`product_id_to_index.json`).

**Index Building → API Server**: Interface is FAISS index file (`product_index.faiss`) and product ID files.

**API Server → Client**: REST API endpoints (`/encode_buyer`, `/retrieve`, `/health`). Pydantic models define request/response schemas.

---

## 8. Error Handling & Edge Cases

### Error Handling Strategy

**Configuration Errors**: `load_config()` raises `FileNotFoundError` if config missing. No fallback; system fails fast.

**Data Loading Errors**: `DataProcessor.load_events()` raises `FileNotFoundError` if CSV missing, `ValueError` if required columns missing. Missing data: `dropna()` removes rows with null critical fields.

**Model Loading Errors**: `EmbeddingEncoder._load_model()` raises exceptions if checkpoint missing or incompatible. Categorical vocab reconstruction: falls back to dummy vocabs if missing, reconstructs from metadata if available.

**API Errors**: FastAPI handles exceptions. `HTTPException(status_code=503)` for uninitialized components. Generic `Exception` caught in endpoints, returned as 500 with error message in `detail`.

**FAISS Errors**: `VectorDatabase` raises `ValueError` if index not built/loaded before retrieval. Dimension mismatch checked on `build_index()`.

### Edge Cases Handled

**Empty Sequences**: Buyer with no interactions → `encode_buyer()` receives empty list → falls back to single-item sequence if available (training uses positive product text as fallback).

**Missing Product Metadata**: Product ID in interactions but not in metadata → categorical features use `<UNK>` (index 0). Text uses empty string → encoded as single space.

**Duplicate Products**: Deduplication in `DataProcessor.load_products()` removes duplicates before embedding generation. Dedup key: normalized title+description+brand.

**Large Sequences**: Truncated to `max_interaction_history` (100). Most recent interactions kept.

**Empty Text**: Product with no title/description → encoded as `" "` (single space) to avoid empty string errors in sentence transformer.

**Missing Categories/Brands**: Optional fields. `None` values handled in categorical embedding lookup (maps to `<UNK>`).

**FAISS Index Size**: `retrieve()` clamps `k` to `index.ntotal` to avoid out-of-bounds.

**Product Not in DataFrame**: API `retrieve()` endpoint returns `"N/A"` for title/description if product ID not found in DataFrame (non-fatal).

### Logging

No structured logging. Print statements used throughout. No log levels, file output, or centralized logging configuration.

**Startup Logging**: API server prints initialization messages (model loaded, index loaded, metadata loaded).

**Training Logging**: Progress bars (`tqdm`), epoch summaries printed to console.

**Error Messages**: Exceptions contain descriptive messages. API returns error details in HTTP response body.

---

## 9. Performance & Scalability Considerations

### Performance Bottlenecks

**Training**:
- Buyer sequence encoding: Optimized via `_encode_buyer_sequences_batched()` (single batch vs. N sequential calls)
- Text encoder: Frozen by default reduces computation
- Batch size: Default 512; memory-bound on GPU

**Inference**:
- Product embedding generation: Batch size 64 (configurable in `generate_embeddings.py`). Linear with catalog size.
- Buyer encoding: Single sequence encoding (real-time). Sequence length limited to 100.
- FAISS search: Exact search (IndexFlatIP) is O(n) where n = catalog size. No approximation used.

**API Server**:
- Product DataFrame lookup: O(n) scan for product_id matching (no index). Consider indexing if catalog > 100K.
- Photo dictionary: O(1) lookup (dict).
- Global state: All artifacts loaded at startup (memory trade-off for latency).

### Optimizations Implemented

**Batch Encoding**: Training batches all sequence texts, single forward pass through ItemTower.

**GPU Utilization**: Automatic device selection based on `config['inference']['device']` and CUDA availability. Tensors moved to device automatically.

**FAISS Normalization**: Embeddings re-normalized on index build (safety check). Query normalized before search.

**Chunked CSV Reading**: `DataProcessor.load_events()` uses `chunksize=100000` to handle large files.

**Product Deduplication**: Runs before embedding generation, reduces catalog size.

### Scalability Limitations

**Catalog Size**: FAISS IndexFlatIP is O(n) search. For catalogs > 1M, consider IndexIVF (approximate search) or sharding.

**Memory**: Product DataFrame loaded in memory (API server). Photo dictionary in memory. For very large catalogs, consider database or caching layer.

**Training Data**: Positive pairs stored in memory (`TwoTowerDataset`). For very large datasets, consider streaming or data sharding.

**Concurrent Requests**: API server handles requests sequentially per process. For high throughput, use multiple workers (`uvicorn --workers N`) or load balancer.

**GPU Memory**: Training batch size 512 may exceed GPU memory on smaller GPUs. Reduce `batch_size` in config.

### Scaling Strategies

**Horizontal Scaling**: API server stateless (after startup). Multiple instances behind load balancer possible. Each instance loads model/index independently.

**FAISS GPU**: Use `faiss-gpu` (conda install) for GPU-accelerated search. Requires CUDA.

**Caching**: Buyer embeddings could be cached (not implemented). Product embeddings pre-computed (already cached).

**Database**: Replace DataFrame with database (PostgreSQL, etc.) for product metadata. Requires code changes in `server.py`.

---

## 10. Security Considerations

### Authentication & Authorization

**None implemented**. API endpoints are publicly accessible. No authentication, API keys, or rate limiting.

**CORS**: Configured to allow all origins (`allow_origins=["*"]`). Comment in code suggests restricting to specific frontend URL in production.

### Data Exposure

**Product Data**: All product metadata (title, description, brand, category) returned in API responses. No filtering or access control.

**Buyer Data**: Buyer IDs and interactions accepted in requests. No validation of buyer ID format or ownership. No PII handling (system assumes buyer IDs are not PII).

**Error Messages**: API returns detailed error messages in responses (e.g., `"Error encoding buyer: ..."`). May leak internal implementation details.

### Input Validation

**Pydantic Models**: Request validation via Pydantic. Invalid fields return 422 with validation errors.

**Interaction Limits**: `max_interactions_per_request: 100` enforced via Pydantic `Field(max_items=100)`.

**K Parameter**: Retrieval `k` limited to 1-100 via `Field(ge=1, le=100)`.

**Product IDs**: No validation of product ID format. Assumes string IDs.

### Sensitive Data Handling

**Checkpoints**: Model files contain no user data. Vocabularies contain product metadata (brands, categories) but no buyer data.

**Logs**: Print statements may output to stdout/stderr. No sensitive data logged currently.

**Environment Variables**: `.env` file not used by core code. If used in deployment, ensure file permissions restrict access.

### Recommendations

- Add API key authentication or OAuth2
- Implement rate limiting (e.g., `slowapi`)
- Restrict CORS origins in production
- Sanitize error messages (don't expose stack traces)
- Validate buyer ID format/ownership if buyer data is sensitive
- Use HTTPS in production (handled by reverse proxy/load balancer)

---

## 11. Development & Debugging Guide

### Local Development Setup

**Prerequisites**: Python 3.8+, virtual environment, CUDA toolkit (optional, for GPU).

**Setup Steps**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt  # CPU version
# OR
pip install -r requirements-gpu.txt  # GPU version (adjust PyTorch index URL for CUDA version)

# Verify installation
python -c "import torch; import sentence_transformers; import faiss; print('OK')"
```

**Data Preparation**: Place `events.csv` and `products.csv` in `data/` directory. Column names normalized internally; see `DataProcessor` for expected formats.

**Configuration**: Copy `configs/config.yaml` (or use default). Adjust paths, batch sizes, device settings as needed.

### Running the System

**Training**:
```bash
python scripts/train.py
```
Output: Checkpoints in `checkpoints/`, best model saved as `best_model.pt`.

**Generate Embeddings**:
```bash
python scripts/generate_embeddings.py
```
Requires: Trained model checkpoint. Output: `outputs/embeddings/product_embeddings.npy`, `product_ids.npy`, `product_id_to_index.json`.

**Build Index**:
```bash
python scripts/build_index.py
```
Requires: Generated embeddings. Output: `outputs/index/product_index.faiss`, `product_ids.npy`, `product_id_to_index.json`.

**Start API Server**:
```bash
python -m src.api.server
# OR
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
Requires: Model checkpoint, FAISS index, product CSV. Access: `http://localhost:8000`.

**Evaluate**:
```bash
python scripts/evaluate.py --k-values 1 5 10 20 --output outputs/eval.json
```
Output: JSON file with metrics.

### Common Pitfalls

**Import Errors**: Scripts modify `sys.path` and change working directory. Run scripts from project root. If issues, check `project_root` path resolution.

**CUDA Out of Memory**: Reduce `batch_size` in config. Check GPU memory: `nvidia-smi`. Use CPU if GPU memory insufficient (`device: "cpu"`).

**Missing Checkpoint**: Training must complete before embedding generation. Checkpoint path: `checkpoints/best_model.pt` (or update config).

**FAISS Index Not Found**: Run `build_index.py` after `generate_embeddings.py`. Index path: `outputs/index/product_index.faiss`.

**Product Metadata Not Set**: `EmbeddingEncoder` requires `set_product_metadata()` before encoding. API server sets this at startup.

**Categorical Vocab Mismatch**: If checkpoint has categorical embeddings but vocabs missing, encoder reconstructs from product metadata. Ensure product metadata matches training data.

**Empty Recommendations**: Check buyer interactions contain valid product IDs. Verify product IDs exist in FAISS index.

**Windows DataLoader**: `num_workers=0` hardcoded in training scripts for Windows. On Linux, can increase for faster loading.

### Debugging Tips

**Model Loading**: Check checkpoint keys: `torch.load('checkpoint.pt', map_location='cpu').keys()`. Verify `model_state_dict`, `brand_vocab`, `category_vocab` exist.

**Embedding Dimensions**: Verify embeddings are 384-dim (or configured dimension). Check normalization: `np.linalg.norm(embedding)` should be ~1.0.

**Sequence Encoding**: Inspect `Trainer._encode_buyer_sequences_batched()` output shapes. Buyer sequences should be `[batch_size, seq_len, 384]` after padding.

**FAISS Search**: Test with known product: encode product text, search index, verify product appears in results.

**API Requests**: Use FastAPI docs (`http://localhost:8000/docs`) for interactive testing. Check request/response schemas.

**Data Quality**: Verify CSV files: check for nulls, encoding issues (UTF-8 for Arabic), column names. Use `pandas.read_csv()` directly to inspect.

**GPU Debugging**: Verify CUDA: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`. Check device placement: `tensor.device`.

**Memory Profiling**: Monitor memory usage during training/inference. Use `torch.cuda.memory_allocated()` for GPU, `psutil` for CPU.

### Testing

**Sanity Checks**: Run `python tests/test_sanity_checks.py` to verify item embeddings (similarity test) and buyer retrieval (requires trained model).

**Unit Tests**: Limited test coverage. `tests/` contains sanity checks only. No unit tests for individual modules.

**Integration Testing**: Manual testing via API endpoints. Evaluation script (`evaluate.py`) provides end-to-end validation.

---

## 12. Extension Points

### Safe Extension Points

**New Aggregation Methods**: Add methods to `BuyerTower` (e.g., `lstm_aggregation()`). Update `forward()` to handle new method. Configuration: add to `config.yaml` `buyer_tower.aggregation_method`.

**Additional Categorical Features**: Extend `ItemTower` to support more categorical embeddings (e.g., price range, seller). Add embedding layers, update projection input dimension, extend `initialize_categorical_embeddings()`.

**Custom Loss Functions**: Create new loss class in `src/training/losses.py`, implement `forward()` method. Update `Trainer.__init__()` to use new loss.

**Evaluation Metrics**: Add functions to `src/evaluation/metrics.py`. Follow existing pattern (function returns float, `Evaluator` class aggregates). Update `Evaluator.evaluate_retrieval()` to compute new metrics.

**API Endpoints**: Add endpoints to `src/api/server.py`. Follow existing pattern: Pydantic request/response models, error handling, global state access.

**Data Processing**: Extend `DataProcessor` with new methods (e.g., filtering, augmentation). Used by training/evaluation scripts.

### Modules to Avoid Modifying Directly

**Checkpoint Format**: Changing checkpoint structure breaks backward compatibility. If modifying, implement migration logic in `EmbeddingEncoder._load_model()`.

**FAISS Index Format**: Index format is FAISS-internal. Use `VectorDatabase` methods (`build_index()`, `save_index()`, `load_index()`) rather than direct FAISS calls.

**TwoTowerModel.forward()**: Legacy method (uses placeholder sequence encoding). Use `forward_simplified()` instead. Do not modify `forward_simplified()` signature without updating `Trainer`.

**DataLoader Configuration**: `num_workers=0` in scripts is intentional (Windows compatibility). Modify only if cross-platform handling added.

**Global State in API Server**: Module-level globals (`encoder`, `vector_db`, etc.) are initialized at startup. Changing initialization order may break startup. Use startup event for initialization logic.

### Extension Patterns

**Adding New Features to Item Tower**:
1. Add feature extraction in `ItemTower.__init__()` (e.g., image encoder, numerical features)
2. Update projection input dimension
3. Modify `forward()` to concatenate new features
4. Update checkpoint saving/loading if needed

**Adding New Metrics**:
1. Implement metric function in `metrics.py` (e.g., `compute_new_metric()`)
2. Add to `Evaluator.evaluate_retrieval()` or create new method
3. Update `evaluate_all()` to include new metric
4. Update evaluation script if needed

**Custom Data Sources**:
1. Extend `DataProcessor` with new loader (e.g., `load_events_from_db()`)
2. Maintain same output format (DataFrame with expected columns)
3. Update scripts to use new loader (or make it configurable)

**API Authentication**:
1. Add dependency in FastAPI (e.g., `get_api_key()`)
2. Add to endpoint decorators: `@app.post("/retrieve", dependencies=[Depends(get_api_key)])`
3. Implement key validation logic

**Database Integration**:
1. Replace `products_df` global with database connection/session
2. Implement product lookup method (replace DataFrame filtering)
3. Update `server.py` startup to initialize database
4. Consider connection pooling, query optimization

### Backward Compatibility

**Configuration Changes**: Adding new config fields with defaults maintains compatibility. Removing fields may break. Use `config.get('key', default)` for optional fields.

**Model Architecture Changes**: Changing embedding dimensions breaks checkpoints. Implement migration or versioning. Changing categorical vocab sizes requires checkpoint retraining.

**API Response Changes**: Adding fields to Pydantic models is backward compatible (clients ignore unknown fields). Removing fields breaks clients. Use versioning (`/v1/`, `/v2/`) for breaking changes.

**Checkpoint Compatibility**: Old checkpoints may lack `brand_vocab`/`category_vocab`. `EmbeddingEncoder` handles this (reconstructs from metadata). For other changes, implement checkpoint migration.

---

## Appendix: Key Constants & Defaults

**Embedding Dimension**: 384 (configurable via `model.embedding_dim`)

**Text Encoder**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)

**Categorical Embedding Dim**: 64 per feature (brand, category)

**Max Interaction History**: 100 interactions (configurable)

**Event Weights**: view=1, add_to_cart=5, purchase=10

**Training Batch Size**: 512 (configurable)

**InfoNCE Temperature**: 0.07 (configurable)

**Negative Samples**: 4 per positive (configurable)

**FAISS Index Type**: IndexFlatIP (exact search, inner product)

**API Max Interactions**: 100 per request (Pydantic validation)

**Embedding Generation Batch Size**: 64 (hardcoded in `generate_embeddings.py`)

**DataLoader Workers**: 0 (Windows compatibility, hardcoded in scripts)

