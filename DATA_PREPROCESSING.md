# Data Preprocessing & Insights

## Overview

This document describes the data preprocessing pipeline and provides insights about the buyer interaction and product catalog data used to train the Two-Tower recommendation model.

---

## Data Sources

### 1. Events Data (`events.csv`)
**Purpose**: Buyer-product interaction events (views, cart additions, purchases)

**Required Columns**:
- `buyer_id` (or `distinct_id`) - Unique buyer identifier
- `product_id` - Unique product identifier
- `event_type` (or `event_name`) - Type of interaction: `view`, `add_to_cart`, `purchase`
- `timestamp` (or `created_at`) - Event timestamp in ISO format

**Event Types & Weights**:
- `view`: Weight = 1 (weak signal)
- `add_to_cart`: Weight = 5 (medium signal)
- `purchase`: Weight = 10 (strong signal)

### 2. Products Data (`products.csv`)
**Purpose**: Product catalog with Arabic text and metadata

**Required Columns**:
- `product_id` (or `id`) - Unique product identifier
- `title` - Product title (Arabic text)
- `description` - Product description (Arabic text)
- `metadata` - JSON string with optional `brand` and `catalog_id` (category)

---

## Preprocessing Pipeline

### Events Data Processing

#### 1. **Column Standardization**
- Maps alternative column names to standard names:
  - `distinct_id` → `buyer_id`
  - `event_name` → `event_type`
  - `created_at` → `timestamp`

#### 2. **Data Cleaning**
- Converts timestamps to datetime format (handles parsing errors gracefully)
- Removes rows with missing critical fields (`buyer_id`, `product_id`, `event_type`)
- Normalizes event type names: lowercase, replace spaces with underscores

#### 3. **Chunked Loading**
- Reads large CSV files in chunks (100K rows per chunk) to handle memory constraints
- Automatically concatenates chunks into single DataFrame

#### 4. **Sequence Creation**
- Groups events by `buyer_id` and sorts by timestamp
- Creates interaction sequences: `(product_id, event_weight, timestamp)`
- Limits sequence length to 100 most recent interactions (configurable)

### Products Data Processing

#### 1. **Column Normalization**
- Maps `id` column to `product_id` if needed

#### 2. **Metadata Extraction**
- Parses JSON `metadata` column to extract:
  - `brand` from `metadata['brand']`
  - `category` from `metadata['catalog_id']`
- Handles missing/invalid JSON gracefully (returns `None`)

#### 3. **Text Combination**
- Combines `title` and `description` into single `text` field
- Handles missing fields: uses title if description missing, description if title missing
- Falls back to empty string if both missing (filtered out later)

#### 4. **Deduplication**
- Removes duplicate products based on normalized content:
  - Normalizes: lowercase, strip whitespace, remove extra spaces
  - Creates deduplication key: `title || description || brand`
  - Keeps first occurrence, removes subsequent duplicates
- **Impact**: Prevents duplicate products in FAISS index

#### 5. **Quality Filtering**
- Removes products with empty text (no title or description)
- Ensures all products have at least minimal text for encoding

---

## Data Insights

### Expected Data Characteristics

#### Events Data
- **Volume**: Typically 100K - 10M+ interaction events
- **Buyer Distribution**: Long-tail distribution (few buyers with many interactions, many with few)
- **Event Distribution**: 
  - Views: ~70-80% of events
  - Add to Cart: ~15-20% of events
  - Purchases: ~5-10% of events
- **Temporal Patterns**: Events clustered by time (sessions, shopping periods)

#### Products Data
- **Catalog Size**: 10K - 500K+ products
- **Text Quality**: 
  - Arabic text in title/description
  - Variable length (short titles, longer descriptions)
  - May contain special characters, emojis, or formatting
- **Metadata Coverage**:
  - Brand: ~60-80% coverage
  - Category: ~70-90% coverage
- **Deduplication Impact**: Typically removes 5-15% of products

### Data Quality Metrics

#### Events Quality
- **Completeness**: 
  - Missing `buyer_id`: Filtered out
  - Missing `product_id`: Filtered out
  - Missing `event_type`: Filtered out
  - Invalid timestamps: Set to `NaT`, kept if other fields valid
- **Consistency**: Event types normalized to lowercase with underscores

#### Products Quality
- **Completeness**:
  - Products without text: Removed (~1-5% typically)
  - Missing brand/category: Handled as `None` (uses `<UNK>` token)
- **Uniqueness**: Deduplication removes exact content duplicates

---

## Key Transformations

### 1. **Event Weighting**
Events are weighted by importance for training:
```python
view → weight = 1
add_to_cart → weight = 5
purchase → weight = 10
```
These weights influence how buyer sequences are aggregated in the Buyer Tower.

### 2. **Text Normalization**
- Arabic text preserved as-is (multilingual encoder handles it)
- Whitespace normalized (multiple spaces → single space)
- Empty strings replaced with single space `" "` (required by sentence transformer)

### 3. **Sequence Truncation**
- Buyer interaction sequences limited to 100 most recent events
- Older interactions discarded (keeps recent behavior patterns)
- Configurable via `model.buyer_tower.max_interaction_history`

### 4. **Categorical Feature Handling**
- Brand and category converted to indices via vocabulary
- Missing values mapped to `<UNK>` token (index 0)
- Vocabularies built dynamically from product metadata

---

## Data Statistics (Example)

After preprocessing, typical dataset characteristics:

| Metric | Value |
|--------|-------|
| **Total Events** | 1M - 10M |
| **Unique Buyers** | 50K - 500K |
| **Unique Products** | 10K - 200K |
| **Avg Interactions/Buyer** | 5 - 20 |
| **Products with Brand** | 60-80% |
| **Products with Category** | 70-90% |
| **Deduplicated Products** | 5-15% reduction |

---

## Preprocessing Output

### For Training
- **Positive Pairs**: List of `(buyer_id, product_id, weight)` tuples
- **Buyer Sequences**: Dictionary mapping `buyer_id` → `[(product_id, weight, timestamp), ...]`
- **Product Metadata**: Dictionary mapping `product_id` → `{text, brand, category, title, description}`

### For Inference
- **Product Embeddings**: Pre-computed 384-dim vectors for all products
- **FAISS Index**: Vector database for fast similarity search
- **Product Metadata**: Loaded for API responses (title, description, brand, category)

---

## Data Quality Considerations

### Common Issues & Solutions

1. **Missing Product Text**
   - **Issue**: Products without title/description
   - **Solution**: Filtered out during preprocessing
   - **Impact**: Reduces catalog size but improves quality

2. **Duplicate Products**
   - **Issue**: Same product with different IDs
   - **Solution**: Content-based deduplication
   - **Impact**: Prevents duplicate recommendations

3. **Invalid Timestamps**
   - **Issue**: Unparseable timestamp strings
   - **Solution**: Set to `NaT`, kept if other fields valid
   - **Impact**: Sequence ordering may be affected

4. **Missing Metadata**
   - **Issue**: Products without brand/category
   - **Solution**: Uses `<UNK>` token in categorical embeddings
   - **Impact**: Model learns to handle missing features

5. **Large File Sizes**
   - **Issue**: Memory constraints with large CSVs
   - **Solution**: Chunked reading (100K rows per chunk)
   - **Impact**: Enables processing of very large datasets

---

## Configuration

Preprocessing behavior controlled via `configs/config.yaml`:

```yaml
data:
  events_path: "data/events.csv"
  products_path: "data/products.csv"

model:
  buyer_tower:
    max_interaction_history: 100  # Sequence length limit

event_weights:
  view: 1
  add_to_cart: 5
  purchase: 10
```

---

## Usage

### Load and Process Data

```python
from src.data.processor import DataProcessor

processor = DataProcessor()

# Load events
events_df = processor.load_events()

# Load products
products_df = processor.load_products()

# Create interaction sequences
sequences = processor.create_interaction_sequences(events_df)

# Create positive pairs for training
positive_pairs = processor.create_positive_pairs(events_df)

# Get product metadata
metadata = processor.get_product_metadata(products_df)
```

---

## Summary

The preprocessing pipeline ensures:
- ✅ **Clean data**: Removes invalid/missing records
- ✅ **Normalized format**: Standardizes column names and event types
- ✅ **Deduplication**: Removes duplicate products
- ✅ **Efficient loading**: Handles large files via chunking
- ✅ **Sequence creation**: Builds buyer interaction sequences with event weights
- ✅ **Metadata extraction**: Parses JSON metadata for categorical features

This preprocessing enables the Two-Tower model to learn effective embeddings from buyer behavior and product content.

