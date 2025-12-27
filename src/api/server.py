"""FastAPI server for real-time inference."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

from src.inference.encoder import EmbeddingEncoder
from src.inference.vector_db import VectorDatabase
from src.utils.config import load_config


# Request/Response models
class Interaction(BaseModel):
    """Single buyer-product interaction."""
    product_id: str
    event_type: str = Field(..., description="Event type: 'view', 'add_to_cart', or 'purchase'")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")


class EncodeBuyerRequest(BaseModel):
    """Request for encoding a buyer."""
    buyer_id: str
    recent_interactions: List[Interaction] = Field(
        ...,
        max_items=100,
        description="Recent buyer interactions (up to 100)"
    )


class EncodeBuyerResponse(BaseModel):
    """Response with buyer embedding."""
    buyer_id: str
    buyer_embedding: List[float]
    dimension: int


class RetrieveRequest(BaseModel):
    """Request for retrieving products."""
    buyer_id: str
    recent_interactions: List[Interaction]
    k: int = Field(10, ge=1, le=100, description="Number of products to retrieve")


class RetrieveResponse(BaseModel):
    """Response with retrieved products."""
    buyer_id: str
    product_ids: List[str]
    scores: List[float]


# Initialize FastAPI app
app = FastAPI(
    title="Two-Tower Recommendation API",
    description="Real-time buyer encoding and product retrieval API",
    version="1.0.0"
)

# Global variables (initialized at startup)
encoder: Optional[EmbeddingEncoder] = None
vector_db: Optional[VectorDatabase] = None
config: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and vector database at startup."""
    global encoder, vector_db, config
    
    config = load_config()
    
    # Load encoder
    model_path = config['inference']['model_checkpoint']
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    encoder = EmbeddingEncoder(
        model_path=model_path,
        config_path="configs/config.yaml"
    )
    
    # Load vector database
    index_dir = Path(config['inference']['index_dir'])
    index_path = index_dir / "product_index.faiss"
    product_ids_path = index_dir / "product_ids.npy"
    mapping_path = index_dir / "product_id_to_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    vector_db = VectorDatabase(embedding_dim=config['model']['embedding_dim'])
    vector_db.load_index(
        str(index_path),
        str(product_ids_path) if product_ids_path.exists() else None,
        str(mapping_path) if mapping_path.exists() else None
    )
    
    # Load product metadata for encoder
    # This should be loaded from a preprocessed file or database
    # For now, we'll assume it's set elsewhere or loaded from products.csv
    print("API server initialized successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Two-Tower Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "encode_buyer": "POST /encode_buyer",
            "retrieve": "POST /retrieve",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "encoder_loaded": encoder is not None,
        "vector_db_loaded": vector_db is not None
    }


@app.post("/encode_buyer", response_model=EncodeBuyerResponse)
async def encode_buyer(request: EncodeBuyerRequest):
    """Encode a buyer from recent interactions.
    
    Args:
        request: Buyer encoding request
        
    Returns:
        Buyer embedding vector
    """
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized")
    
    try:
        # Convert interactions to format expected by encoder
        interactions = [
            {
                "product_id": interaction.product_id,
                "event_type": interaction.event_type,
                "timestamp": interaction.timestamp
            }
            for interaction in request.recent_interactions
        ]
        
        # Encode buyer
        buyer_embedding = encoder.encode_buyer(interactions)
        
        return EncodeBuyerResponse(
            buyer_id=request.buyer_id,
            buyer_embedding=buyer_embedding.tolist(),
            dimension=len(buyer_embedding)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding buyer: {str(e)}")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Retrieve top-K products for a buyer.
    
    Args:
        request: Retrieval request
        
    Returns:
        Retrieved product IDs and scores
    """
    if encoder is None or vector_db is None:
        raise HTTPException(status_code=503, detail="Encoder or vector database not initialized")
    
    try:
        # Encode buyer
        interactions = [
            {
                "product_id": interaction.product_id,
                "event_type": interaction.event_type,
                "timestamp": interaction.timestamp
            }
            for interaction in request.recent_interactions
        ]
        
        buyer_embedding = encoder.encode_buyer(interactions)
        
        # Retrieve products
        results = vector_db.retrieve(buyer_embedding, k=request.k)
        
        product_ids = [pid for pid, _ in results]
        scores = [score for _, score in results]
        
        return RetrieveResponse(
            buyer_id=request.buyer_id,
            product_ids=product_ids,
            scores=scores
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving products: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        "src.api.server:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=False
    )

