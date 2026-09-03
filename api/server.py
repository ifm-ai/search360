#!/usr/bin/env python3
"""
FastAPI service for FAISS-based passage retrieval.
"""

import argparse
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from api.search import load_search_system, embed_query, get_passage, get_document, get_passages_parallel, _resolve_k_fetch
from api.utils import rank_documents_by_occurrence
import numpy as np


# Global system state (loaded on startup)
SEARCH_SYSTEM = None
MAX_K = 100


def get_default_args():
    """Get default arguments from environment variables or hardcoded defaults."""

    class Args:
        def __init__(self):
            self.index_path = os.getenv(
                "INDEX_PATH",
                "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/index_faiss_msmarco/final_index.faiss",
            )
            self.output_dir = os.getenv(
                "OUTPUT_DIR",
                "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data",
            )
            self.passages_dir = os.getenv(
                "PASSAGES_DIR",
                "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/passages",
            )
            self.documents_dir = os.getenv(
                "DOCUMENTS_DIR",
                "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/raw_high_data",
            )
            self.model_name = os.getenv("MODEL_NAME", "facebook/contriever-msmarco")
            self.nprobe = int(os.getenv("NPROBE", "256"))
            self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
            self.load_reranker = os.getenv("LOAD_RERANKER", "true").lower() == "true"

    return Args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search system on startup, cleanup on shutdown."""
    global SEARCH_SYSTEM
    print("Loading search system...")

    # Get args from app state if available, otherwise use defaults
    args = getattr(app.state, "args", None) or get_default_args()
    SEARCH_SYSTEM = load_search_system(args)

    print(f"✓ System loaded: {SEARCH_SYSTEM['index'].ntotal:,} passages")

    yield

    # Cleanup on shutdown (if needed)
    print("Shutting down search system...")


# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(10, description="Number of results to return", ge=1, le=MAX_K)
    return_fulltext: bool = Field(False, description="Return full source documents")
    rerank: bool = Field(True, description="Use re-ranking to improve results")
    initial_k: int = Field(25, description="Number of passages to retrieve before re-ranking", ge=1, le=1000)
    min_words: Optional[int] = Field(None, description="Minimum words required in passage (filter short passages)")


class PassageResult(BaseModel):
    rank: int
    score: float
    passage_id: int
    passage_text: str
    passage_file: str
    passage_position: int
    doc_id: int
    rerank_score: Optional[float] = None


class DocumentResult(BaseModel):
    rank: int
    doc_id: int
    occurrences: int
    doc_text: str
    doc_file: str
    doc_position: int


class SearchResponse(BaseModel):
    query: str
    results: List[PassageResult]
    documents: Optional[List[DocumentResult]] = None


class GetDocumentRequest(BaseModel):
    doc_id: int = Field(..., description="Document ID to retrieve")


class DocumentResponse(BaseModel):
    doc_id: int
    doc_text: str
    doc_file: str
    doc_position: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


class StatsResponse(BaseModel):
    total_passages: int
    total_documents: int
    num_passage_files: int
    num_document_files: int


# Create FastAPI app
app = FastAPI(
    title="FAISS Passage Retrieval API",
    description="Search engine for retrieving passages and documents from FAISS index",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if SEARCH_SYSTEM is not None else "not_ready",
        "index_loaded": SEARCH_SYSTEM is not None,
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    if SEARCH_SYSTEM is None:
        raise HTTPException(status_code=503, detail="System not loaded")

    return {
        "total_passages": int(SEARCH_SYSTEM["index"].ntotal),
        "total_documents": len(SEARCH_SYSTEM["doc_id_to_file_id"]),
        "num_passage_files": len(SEARCH_SYSTEM["passage_filenames"]),
        "num_document_files": len(SEARCH_SYSTEM["doc_filenames"]),
    }


@app.get("/metrics")
async def metrics():
    """Stub endpoint to prevent 404s from metrics scrapers."""
    return {}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for passages and optionally return source documents.

    Args:
        query: Search query string
        k: Number of results to return (max 100)
        return_fulltext: If true, also return ranked source documents

    Returns:
        Search results with passages and optionally documents
    """
    if SEARCH_SYSTEM is None:
        raise HTTPException(status_code=503, detail="System not loaded")

    # Validate query
    if not request.query or not request.query.strip():
        return {
            "query": request.query,
            "results": [],
            "documents": [] if request.return_fulltext else None,
        }

    # Create query embedding
    query_embedding = embed_query(
        request.query, SEARCH_SYSTEM["model"], SEARCH_SYSTEM["tokenizer"]
    )

    # Calculate fetch size - scale up if min_words filter is applied
    base_k = request.initial_k if request.rerank else request.k
    fetch_k = _resolve_k_fetch(base_k, request.min_words)

    # Search index
    scores, passage_ids = SEARCH_SYSTEM["index"].search(
        query_embedding.astype(np.float32), fetch_k
    )

    # Retrieve passages in parallel (with optional min_words filter)
    raw_passages = get_passages_parallel(
        passage_ids[0].tolist(),
        SEARCH_SYSTEM,
        min_words=request.min_words
    )

    # Build results with scores
    pid_to_score = {int(pid): float(scores[0][i]) for i, pid in enumerate(passage_ids[0])}

    passage_results = []
    for passage, psg_filename, psg_position, passage_id in raw_passages:
        score = pid_to_score.get(passage_id, 0.0)
        doc_id = int(SEARCH_SYSTEM["passage_to_doc_id"][passage_id])

        result = {
            "rank": len(passage_results) + 1,
            "score": score,
            "passage_id": int(passage_id),
            "passage_text": passage["text"],
            "passage_file": psg_filename,
            "passage_position": int(psg_position),
            "doc_id": doc_id,
        }

        passage_results.append(result)

        # Stop if we have enough results
        if len(passage_results) >= base_k:
            break

    # Sort by score and update ranks
    passage_results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(passage_results):
        r["rank"] = i + 1

    # Re-rank if requested and model is loaded
    if request.rerank and SEARCH_SYSTEM["reranker_model"] is not None:
        from api.search import rerank_passages
        passage_results = rerank_passages(request.query, passage_results, SEARCH_SYSTEM)
        # Keep only top k after re-ranking
        passage_results = passage_results[:request.k]

    # Fetch full documents if requested (after re-ranking)
    if request.return_fulltext:
        for result in passage_results:
            doc_id = result["doc_id"]
            document, doc_filename, doc_position = get_document(doc_id, SEARCH_SYSTEM)
            result["doc_text"] = document.get("text", "")
            result["doc_file"] = doc_filename
            result["doc_position"] = int(doc_position)

    # Prepare response
    response = {"query": request.query, "results": passage_results}

    # Rank documents if requested
    if request.return_fulltext:
        ranked_documents = rank_documents_by_occurrence(passage_results)
        response["documents"] = ranked_documents

    return response


@app.post("/get_document", response_model=DocumentResponse)
async def get_document_by_id(request: GetDocumentRequest):
    """
    Get full document by document ID.

    Args:
        doc_id: Document ID to retrieve

    Returns:
        Document with full text and metadata
    """
    if SEARCH_SYSTEM is None:
        raise HTTPException(status_code=503, detail="System not loaded")

    # Validate doc_id
    if request.doc_id < 0 or request.doc_id >= len(SEARCH_SYSTEM["doc_id_to_file_id"]):
        raise HTTPException(
            status_code=404, detail=f"Document ID {request.doc_id} not found"
        )

    # Get document
    document, doc_filename, doc_position = get_document(request.doc_id, SEARCH_SYSTEM)

    return {
        "doc_id": request.doc_id,
        "doc_text": document.get("text", ""),
        "doc_file": doc_filename,
        "doc_position": int(doc_position),
    }


def main():
    parser = argparse.ArgumentParser(description="FastAPI service for FAISS search")

    parser.add_argument(
        "--index_path",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/index_faiss_msmarco/final_index.faiss",
        help="Path to FAISS index",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data",
        help="Directory containing mappings",
    )

    parser.add_argument(
        "--passages_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/passages",
        help="Directory containing passage JSONL files",
    )

    parser.add_argument(
        "--documents_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_high_quality_data/raw_high_data",
        help="Directory containing document JSONL files",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/contriever-msmarco",
        help="HuggingFace model name (MS MARCO fine-tuned)",
    )

    parser.add_argument(
        "--nprobe", type=int, default=256, help="Number of clusters to probe (ds-serve default)"
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU(s) for FAISS search",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    parser.add_argument("--port", type=int, default=8000, help="Server port")

    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    parser.add_argument(
        "--load_reranker",
        action="store_true",
        default=True,
        help="Load re-ranker model for improved results"
    )

    args = parser.parse_args()

    # Store args in app state for startup event
    app.state.args = args

    # Print configuration
    print("=" * 80)
    print("FAISS API SERVER")
    print("=" * 80)
    print(f"Index: {args.index_path}")
    print(f"Using GPU: {args.use_gpu}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    # Run server
    uvicorn.run("api.server:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
