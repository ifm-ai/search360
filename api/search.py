#!/usr/bin/env python3
"""
Interactive search interface for FAISS index.
Retrieves relevant passages given user queries.
"""

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

# Configure FAISS/BLAS thread count (default: 128)
FAISS_THREADS = int(os.environ.get("FAISS_OMP_THREADS", "128"))
try:
    faiss.omp_set_num_threads(FAISS_THREADS)
except Exception:
    pass
os.environ["OMP_NUM_THREADS"] = str(FAISS_THREADS)
os.environ["MKL_NUM_THREADS"] = str(FAISS_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def mean_pooling(token_embeddings, mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def embed_query(query, model, tokenizer, max_length=128):
    """Create embedding for a query using Contriever."""
    encoding = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Move to GPU if available
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = mean_pooling(outputs[0], inputs["attention_mask"])

    return embedding.cpu().numpy()


def load_search_system(args):
    """Load FAISS index, mappings, and model."""
    print("=" * 80)
    print("LOADING SEARCH SYSTEM")
    print("=" * 80)

    # Load FAISS index
    print(f"\nLoading FAISS index from {args.index_path}...")
    index = faiss.read_index(args.index_path)
    index.nprobe = args.nprobe
    print(f"✓ Index loaded: {index.ntotal:,} passages")

    # Enable direct map for index reconstruction (useful for reranking)
    try:
        if hasattr(index, "make_direct_map"):
            index.make_direct_map()
            print(f"✓ Direct map enabled on index")
        else:
            # Try to extract inner IVF and enable direct map
            try:
                ivf = faiss.extract_index_ivf(index)
                ivf.make_direct_map()
                print(f"✓ Direct map enabled via extract_index_ivf()")
            except Exception:
                print(f"⚠ Could not enable direct map (reconstruct may not work)")
    except Exception as e:
        print(f"⚠ Failed to enable direct map: {e}")

    # Move index to GPU(s) if available
    if args.use_gpu and faiss.get_num_gpus() > 0:
        print(f"\nMoving index to GPU...")
        print(f"Available GPUs: {faiss.get_num_gpus()}")

        if faiss.get_num_gpus() > 1:
            # Use multiple GPUs
            print(f"Using {faiss.get_num_gpus()} GPUs with sharding")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Shard the index across GPUs
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        else:
            # Use single GPU
            print(f"Using single GPU")
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        print(f"✓ Index moved to GPU(s)")
    else:
        print(f"Using CPU for search (use --use_gpu to enable GPU)")

    # Load passage mappings (using memory mapping for large arrays)
    mappings_dir = Path(args.output_dir) / "mappings_passages"
    print(f"\nLoading passage mappings from {mappings_dir}...")

    passage_filenames = np.load(
        mappings_dir / "passage_filenames.npy", allow_pickle=True
    )
    # Use memory mapping for large arrays to reduce RAM usage
    passage_id_to_file_id = np.load(mappings_dir / "passage_id_to_file_id.npy", mmap_mode="r")
    passage_pos_id_array = np.load(mappings_dir / "passage_pos_id_array.npy", mmap_mode="r")
    passage_to_doc_id = np.load(mappings_dir / "passage_to_doc_id.npy", mmap_mode="r")

    print(f"✓ Loaded passage mappings (memory-mapped)")
    print(f"  - {len(passage_filenames)} passage files")
    print(f"  - {len(passage_id_to_file_id):,} passage mappings")

    # Load document mappings (using memory mapping for large arrays)
    doc_mappings_dir = Path(args.output_dir) / "index"
    print(f"\nLoading document mappings from {doc_mappings_dir}...")

    doc_filenames = np.load(doc_mappings_dir / "doc_filenames.npy", allow_pickle=True)
    # Use memory mapping for large arrays to reduce RAM usage
    doc_id_to_file_id = np.load(doc_mappings_dir / "doc_id_to_file_id.npy", mmap_mode="r")
    doc_pos_id_array = np.load(doc_mappings_dir / "doc_pos_id_array.npy", mmap_mode="r")

    print(f"✓ Loaded document mappings (memory-mapped)")
    print(f"  - {len(doc_filenames)} document files")
    print(f"  - {len(doc_id_to_file_id):,} documents")

    # Load Contriever model
    print(f"\nLoading embedding model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"✓ Embedding model loaded on GPU")
    else:
        print(f"✓ Embedding model loaded on CPU")

    model.eval()

    # Load re-ranker model
    reranker_model = None
    reranker_tokenizer = None

    if hasattr(args, 'load_reranker') and args.load_reranker:
        reranker_name = "jinaai/jina-reranker-v3"
        print(f"\nLoading re-ranker model: {reranker_name}...")
        reranker_model = AutoModel.from_pretrained(
            reranker_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            tie_word_embeddings=False
        )

        if torch.cuda.is_available():
            reranker_model = reranker_model.to("cuda")
            print(f"✓ Re-ranker model loaded on GPU")
        else:
            print(f"✓ Re-ranker model loaded on CPU")

        reranker_model.eval()

    return {
        "index": index,
        "passage_filenames": passage_filenames,
        "passage_id_to_file_id": passage_id_to_file_id,
        "passage_pos_id_array": passage_pos_id_array,
        "passage_to_doc_id": passage_to_doc_id,
        "doc_filenames": doc_filenames,
        "doc_id_to_file_id": doc_id_to_file_id,
        "doc_pos_id_array": doc_pos_id_array,
        "passages_dir": Path(args.passages_dir),
        "documents_dir": Path(args.documents_dir),
        "model": model,
        "tokenizer": tokenizer,
        "reranker_model": reranker_model,
        "reranker_tokenizer": reranker_tokenizer,
    }


def get_passage(passage_id, system):
    """Retrieve passage text and metadata given passage_id."""
    # Get passage file and position
    file_id = int(system["passage_id_to_file_id"][passage_id])
    filename = system["passage_filenames"][file_id]
    position = int(system["passage_pos_id_array"][passage_id])

    # Read passage from file
    filepath = system["passages_dir"] / filename
    with open(filepath, "rb") as f:
        f.seek(position)
        line = f.readline().decode('utf-8', errors='strict')
        passage = json.loads(line)

    return passage, filename, position


# Number of threads for parallel passage retrieval
PASSAGE_READ_THREADS = int(os.environ.get("PASSAGE_READ_THREADS", "24"))


def get_passages_parallel(passage_ids, system, min_words=None):
    """Retrieve multiple passages in parallel.

    Args:
        passage_ids: List of passage IDs to retrieve
        system: System dict with mappings
        min_words: Minimum number of words required (filter short passages)

    Returns:
        List of (passage, filename, position) tuples
    """
    def fetch_single_passage(passage_id):
        try:
            file_id = int(system["passage_id_to_file_id"][passage_id])
            filename = system["passage_filenames"][file_id]
            position = int(system["passage_pos_id_array"][passage_id])

            filepath = system["passages_dir"] / filename
            with open(filepath, "rb") as f:
                f.seek(position)
                line = f.readline().decode('utf-8', errors='strict')
                passage = json.loads(line)

            return (passage, filename, position, passage_id)
        except Exception as e:
            print(f"[WARN] Failed to read passage_id={passage_id}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=PASSAGE_READ_THREADS) as pool:
        futures = [pool.submit(fetch_single_passage, pid) for pid in passage_ids]
        for fut in futures:
            try:
                result = fut.result()
                if result is not None:
                    passage, filename, position, pid = result
                    # Apply minimum words filter if specified
                    if min_words is not None and min_words > 0:
                        text = (passage.get("text") or "").strip()
                        if len(text.split()) < min_words:
                            continue
                    results.append((passage, filename, position, pid))
            except Exception as e:
                print(f"[WARN] Failed to get passage result: {e}")

    return results


def get_document(doc_id, system):
    """Retrieve source document given doc_id."""
    # Get document file and position
    file_id = system["doc_id_to_file_id"][doc_id]
    filename = system["doc_filenames"][file_id]
    position = system["doc_pos_id_array"][doc_id]

    # Read document from file
    filepath = system["documents_dir"] / filename
    with open(filepath, "r") as f:
        f.seek(position)
        document = json.loads(f.readline())

    return document, filename, position


def rerank_passages(query, passages, system):
    """Re-rank passages using Jina re-ranker v3.

    Args:
        query: Search query string
        passages: List of dicts with 'passage_text' and other metadata
        system: System dict containing reranker_model

    Returns:
        Re-ranked list of passages with added 'rerank_score' field
    """
    if system["reranker_model"] is None:
        # Re-ranker not loaded, return passages as-is
        return passages

    # Extract document texts for reranking
    documents = [p["passage_text"] for p in passages]

    # Use Jina reranker's built-in rerank method
    results = system["reranker_model"].rerank(query, documents, top_n=len(documents))

    # Create a mapping from document text to rerank score
    # Results are sorted by relevance, so we need to map back
    doc_to_score = {r["document"]: r["relevance_score"] for r in results}

    # Add rerank scores to passages
    for passage in passages:
        passage["rerank_score"] = float(doc_to_score.get(passage["passage_text"], 0.0))

    # Sort by rerank score (descending)
    reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)

    # Update ranks
    for i, passage in enumerate(reranked):
        passage["rank"] = i + 1

    return reranked


import math


def _resolve_k_fetch(requested_k: int, min_words: int = None, hard_cap: int = 256) -> int:
    """Calculate how many results to fetch when min_words filter is applied.

    When filtering short passages, we need to fetch more to ensure we have enough
    results after filtering.
    """
    base = max(1, int(requested_k))
    if min_words is None or min_words <= 0:
        return base
    # Scale factor based on min_words threshold
    factor = 1 + min(4, max(1, math.ceil(min_words / 50)))
    return min(hard_cap, base * factor)


def search(query, system, k=5, rerank=False, initial_k=25, min_words=None):
    """Search for top-k relevant passages.

    Args:
        query: Search query string
        system: System dict with index, models, etc.
        k: Number of final results to return
        rerank: Whether to use re-ranking (default: False)
        initial_k: Number of passages to retrieve before re-ranking (default: 25)
        min_words: Minimum words required in passage (filter short passages)

    Returns:
        List of search results (re-ranked if rerank=True)
    """
    # Create query embedding
    query_embedding = embed_query(query, system["model"], system["tokenizer"])

    # Calculate how many to fetch - scale up if min_words filter is applied
    base_k = initial_k if rerank else k
    fetch_k = _resolve_k_fetch(base_k, min_words)

    # Search index
    scores, passage_ids = system["index"].search(query_embedding.astype(np.float32), fetch_k)

    # Retrieve passages in parallel
    passage_results = get_passages_parallel(
        passage_ids[0].tolist(),
        system,
        min_words=min_words
    )

    # Build results with scores
    results = []
    # Create a mapping of passage_id to score
    pid_to_score = {int(pid): float(scores[0][i]) for i, pid in enumerate(passage_ids[0])}

    for passage, psg_filename, psg_position, passage_id in passage_results:
        score = pid_to_score.get(passage_id, 0.0)

        # Get source document
        doc_id = int(system["passage_to_doc_id"][passage_id])
        document, doc_filename, doc_position = get_document(doc_id, system)

        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "passage_id": int(passage_id),
                "passage_text": passage["text"],
                "passage_file": psg_filename,
                "passage_position": int(psg_position),
                "doc_id": doc_id,
                "doc_text": document.get("text", ""),
                "doc_file": doc_filename,
                "doc_position": int(doc_position),
            }
        )

        # Stop if we have enough results
        if len(results) >= base_k:
            break

    # Sort by score (highest first) and update ranks
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Re-rank if requested
    if rerank and system["reranker_model"] is not None:
        results = rerank_passages(query, results, system)
        # Keep only top k after re-ranking
        results = results[:k]

    return results


def print_results(query, results):
    """Pretty print search results."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    for result in results:
        print(f"\n[{result['rank']}] Score: {result['score']:.4f}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"\n--- PASSAGE (FULL) ---")
        print(result["passage_text"])
        print(f"\n--- SOURCE DOCUMENT (FULL) ---")
        print(f"Doc ID: {result['doc_id']}")
        print(f"File: {result['doc_file']}")
        print(result["doc_text"])
        print(f"\n--- METADATA ---")
        print(f"Passage file: {result['passage_file']}")
        print(f"Passage position: {result['passage_position']}")
        print(f"Doc position: {result['doc_position']}")
        print("-" * 80)


def interactive_search(system, k=5):
    """Run interactive search loop."""
    print("\n" + "=" * 80)
    print("INTERACTIVE SEARCH")
    print("=" * 80)
    print("Enter your query (or 'quit' to exit)")
    print("=" * 80)

    while True:
        try:
            query = input("\n🔍 Query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not query:
                print("Please enter a query.")
                continue

            # Search
            results = search(query, system, k=k)

            # Display results
            print_results(query, results)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()


def main(args):
    # Load system
    system = load_search_system(args)

    print("\n" + "=" * 80)
    print("✅ SYSTEM READY")
    print("=" * 80)
    print(f"Index size: {system['index'].ntotal:,} passages")
    print(f"Top-k results: {args.k}")
    print(f"nprobe: {args.nprobe}")

    # Start interactive search
    interactive_search(system, k=args.k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive search for FAISS index")

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
        help="Output directory containing mappings",
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
        help="HuggingFace model name for query encoding",
    )

    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to retrieve"
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=256,
        help="Number of clusters to probe during search (ds-serve default)",
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU(s) for FAISS search",
    )

    args = parser.parse_args()

    main(args)
