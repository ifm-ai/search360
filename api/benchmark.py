#!/usr/bin/env python3
"""
Benchmark script for FAISS search system.
Tests 10 queries and logs detailed timing metrics.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

from api.search import load_search_system, embed_query, get_passage, get_document


# 10 benchmark test queries
BENCHMARK_QUERIES = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What are the main causes of climate change?",
    "Explain quantum entanglement",
    "What is machine learning?",
    "How do vaccines work?",
    "What is the theory of relativity?",
    "Describe the water cycle",
    "What causes earthquakes?",
    "How does DNA replication occur?",
]


def benchmark_single_query(query, system, k=10, num_runs=3):
    """
    Benchmark a single query with detailed timing breakdown.
    Runs query multiple times and averages results.
    """
    embedding_times = []
    search_times = []
    retrieval_times = []
    total_times = []

    results = None  # Store results from first run

    for run in range(num_runs):
        run_start = time.perf_counter()

        # 1. Query embedding time
        embed_start = time.perf_counter()
        query_embedding = embed_query(query, system["model"], system["tokenizer"])
        embed_time = time.perf_counter() - embed_start
        embedding_times.append(embed_time)

        # 2. FAISS search time
        search_start = time.perf_counter()
        scores, passage_ids = system["index"].search(
            query_embedding.astype(np.float32), k
        )
        search_time = time.perf_counter() - search_start
        search_times.append(search_time)

        # 3. Passage/document retrieval time
        retrieval_start = time.perf_counter()
        run_results = []
        for i, passage_id in enumerate(passage_ids[0]):
            score = scores[0][i]

            # Get passage
            passage, psg_filename, psg_position = get_passage(passage_id, system)

            # Get source document
            doc_id = system["passage_to_doc_id"][passage_id]
            document, doc_filename, doc_position = get_document(doc_id, system)

            run_results.append(
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

        retrieval_time = time.perf_counter() - retrieval_start
        retrieval_times.append(retrieval_time)

        total_time = time.perf_counter() - run_start
        total_times.append(total_time)

        # Store results from first run
        if run == 0:
            results = run_results

    # Calculate averages
    return {
        "query": query,
        "num_runs": num_runs,
        "timing": {
            "embedding_time": {
                "avg": np.mean(embedding_times),
                "min": np.min(embedding_times),
                "max": np.max(embedding_times),
            },
            "search_time": {
                "avg": np.mean(search_times),
                "min": np.min(search_times),
                "max": np.max(search_times),
            },
            "retrieval_time": {
                "avg": np.mean(retrieval_times),
                "min": np.min(retrieval_times),
                "max": np.max(retrieval_times),
            },
            "total_time": {
                "avg": np.mean(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
            },
        },
        "results": results,  # Results from first run
    }


def run_benchmark(system, k=10, num_runs=3):
    """Run benchmark on all test queries."""
    print("=" * 80)
    print("RUNNING BENCHMARK")
    print("=" * 80)
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print(f"Results per query (k): {k}")
    print(f"Runs per query: {num_runs}")
    print("=" * 80)

    all_query_results = []

    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n[{i}/{len(BENCHMARK_QUERIES)}] Benchmarking: {query[:50]}...")

        query_result = benchmark_single_query(query, system, k=k, num_runs=num_runs)
        all_query_results.append(query_result)

        # Print summary for this query
        timing = query_result["timing"]
        print(f"  Total time (avg): {timing['total_time']['avg']:.4f}s")
        print(f"    - Embedding: {timing['embedding_time']['avg']:.4f}s")
        print(f"    - Search: {timing['search_time']['avg']:.4f}s")
        print(f"    - Retrieval: {timing['retrieval_time']['avg']:.4f}s")

    # Calculate overall statistics
    all_total_times = [q["timing"]["total_time"]["avg"] for q in all_query_results]
    all_embedding_times = [
        q["timing"]["embedding_time"]["avg"] for q in all_query_results
    ]
    all_search_times = [q["timing"]["search_time"]["avg"] for q in all_query_results]
    all_retrieval_times = [
        q["timing"]["retrieval_time"]["avg"] for q in all_query_results
    ]

    overall_stats = {
        "total_queries": len(BENCHMARK_QUERIES),
        "k": k,
        "num_runs_per_query": num_runs,
        "total_time": sum(all_total_times),
        "avg_time_per_query": np.mean(all_total_times),
        "min_time_per_query": np.min(all_total_times),
        "max_time_per_query": np.max(all_total_times),
        "breakdown": {
            "embedding": {
                "avg": np.mean(all_embedding_times),
                "min": np.min(all_embedding_times),
                "max": np.max(all_embedding_times),
            },
            "search": {
                "avg": np.mean(all_search_times),
                "min": np.min(all_search_times),
                "max": np.max(all_search_times),
            },
            "retrieval": {
                "avg": np.mean(all_retrieval_times),
                "min": np.min(all_retrieval_times),
                "max": np.max(all_retrieval_times),
            },
        },
    }

    return {
        "benchmark_date": datetime.now().isoformat(),
        "overall_stats": overall_stats,
        "queries": all_query_results,
    }


def save_results(results, output_dir):
    """Save benchmark results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def print_summary(results):
    """Print benchmark summary."""
    stats = results["overall_stats"]

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Benchmark Date: {results['benchmark_date']}")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Results per Query (k): {stats['k']}")
    print(f"Runs per Query: {stats['num_runs_per_query']}")
    print(f"\nTotal Time: {stats['total_time']:.4f}s")
    print(f"Average Time per Query: {stats['avg_time_per_query']:.4f}s")
    print(f"Min Time per Query: {stats['min_time_per_query']:.4f}s")
    print(f"Max Time per Query: {stats['max_time_per_query']:.4f}s")

    print(f"\nTime Breakdown (averages):")
    print(
        f"  Embedding: {stats['breakdown']['embedding']['avg']:.4f}s ({stats['breakdown']['embedding']['avg']/stats['avg_time_per_query']*100:.1f}%)"
    )
    print(
        f"  Search:    {stats['breakdown']['search']['avg']:.4f}s ({stats['breakdown']['search']['avg']/stats['avg_time_per_query']*100:.1f}%)"
    )
    print(
        f"  Retrieval: {stats['breakdown']['retrieval']['avg']:.4f}s ({stats['breakdown']['retrieval']['avg']/stats['avg_time_per_query']*100:.1f}%)"
    )
    print("=" * 80)


def main(args):
    # Load system
    print("Loading search system...")
    system = load_search_system(args)

    print("\n" + "=" * 80)
    print("✅ SYSTEM READY")
    print("=" * 80)
    print(f"Index size: {system['index'].ntotal:,} passages")
    print(f"Using GPU: {args.use_gpu}")

    # Run benchmark
    results = run_benchmark(system, k=args.k, num_runs=args.num_runs)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS search system")

    parser.add_argument(
        "--index_path",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/index_faiss/final_index.faiss",
        help="Path to FAISS index",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save benchmark results",
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
        "--k", type=int, default=10, help="Number of results to retrieve per query"
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

    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to run each query (for averaging)",
    )

    # Note: output_dir is now different from search_index.py
    # We need to add the mappings path
    parser.add_argument(
        "--mappings_output_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs",
        help="Directory containing mapping files (mappings_passages/, index/)",
    )

    args = parser.parse_args()

    # Hack: add output_dir attribute for load_search_system compatibility
    args.output_dir = args.mappings_output_dir
    output_results_dir = args.output_dir
    args.output_dir = args.mappings_output_dir

    main(args)
