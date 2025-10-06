#!/usr/bin/env python3
"""
Interactive search interface for FAISS index.
Retrieves relevant passages given user queries.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


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

    # Load passage mappings
    mappings_dir = Path(args.output_dir) / "mappings_passages"
    print(f"\nLoading passage mappings from {mappings_dir}...")

    passage_filenames = np.load(
        mappings_dir / "passage_filenames.npy", allow_pickle=True
    )
    passage_id_to_file_id = np.load(mappings_dir / "passage_id_to_file_id.npy")
    passage_pos_id_array = np.load(mappings_dir / "passage_pos_id_array.npy")
    passage_to_doc_id = np.load(mappings_dir / "passage_to_doc_id.npy")

    print(f"✓ Loaded passage mappings")
    print(f"  - {len(passage_filenames)} passage files")
    print(f"  - {len(passage_id_to_file_id):,} passage mappings")

    # Load document mappings
    doc_mappings_dir = Path(args.output_dir) / "index"
    print(f"\nLoading document mappings from {doc_mappings_dir}...")

    doc_filenames = np.load(doc_mappings_dir / "doc_filenames.npy", allow_pickle=True)
    doc_id_to_file_id = np.load(doc_mappings_dir / "doc_id_to_file_id.npy")
    doc_pos_id_array = np.load(doc_mappings_dir / "doc_pos_id_array.npy")

    print(f"✓ Loaded document mappings")
    print(f"  - {len(doc_filenames)} document files")
    print(f"  - {len(doc_id_to_file_id):,} documents")

    # Load Contriever model
    print(f"\nLoading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"✓ Model loaded on GPU")
    else:
        print(f"✓ Model loaded on CPU")

    model.eval()

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
    }


def get_passage(passage_id, system):
    """Retrieve passage text and metadata given passage_id."""
    # Get passage file and position
    file_id = system["passage_id_to_file_id"][passage_id]
    filename = system["passage_filenames"][file_id]
    position = system["passage_pos_id_array"][passage_id]

    # Read passage from file
    filepath = system["passages_dir"] / filename
    with open(filepath, "r") as f:
        f.seek(position)
        passage = json.loads(f.readline())

    return passage, filename, position


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


def search(query, system, k=5):
    """Search for top-k relevant passages."""
    # Create query embedding
    query_embedding = embed_query(query, system["model"], system["tokenizer"])

    # Search index
    scores, passage_ids = system["index"].search(query_embedding.astype(np.float32), k)

    # Retrieve passages and documents
    results = []
    for i, passage_id in enumerate(passage_ids[0]):
        score = scores[0][i]

        # Get passage
        passage, psg_filename, psg_position = get_passage(passage_id, system)

        # Get source document
        doc_id = system["passage_to_doc_id"][passage_id]
        document, doc_filename, doc_position = get_document(doc_id, system)

        results.append(
            {
                "rank": i + 1,
                "score": float(score),
                "passage_id": int(passage_id),
                "passage_text": passage["text"],
                "passage_file": psg_filename,
                "passage_position": int(psg_position),
                "doc_id": int(doc_id),
                "doc_text": document.get("text", ""),
                "doc_file": doc_filename,
                "doc_position": int(doc_position),
            }
        )

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
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/index_faiss/final_index.faiss",
        help="Path to FAISS index",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs",
        help="Output directory containing mappings",
    )

    parser.add_argument(
        "--passages_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/passages",
        help="Directory containing passage JSONL files",
    )

    parser.add_argument(
        "--documents_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/documents_jsonl",
        help="Directory containing document JSONL files",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/contriever",
        help="HuggingFace model name for query encoding",
    )

    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to retrieve"
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=2048,
        help="Number of clusters to probe during search",
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU(s) for FAISS search",
    )

    args = parser.parse_args()

    main(args)
