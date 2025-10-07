#!/usr/bin/env python3
"""
Load testing script for FAISS search API.
Tests with varying concurrency levels and generates detailed performance metrics.

python api/load_test.py --skip_fulltext --num_requests 1000 --concurrency_levels "8,16,32" --k_values "5,10"
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import random

import aiohttp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Diverse test queries (mix of topics)
TEST_QUERIES = [
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
    "Explain neural networks",
    "What is blockchain technology?",
    "How does the stock market work?",
    "What is artificial intelligence?",
    "Explain genetic engineering",
    "How does the immune system work?",
    "What is dark matter?",
    "Describe the nitrogen cycle",
    "What causes inflation?",
    "How do computers process information?",
    "What is quantum computing?",
    "Explain protein folding",
    "How does solar energy work?",
    "What is the carbon cycle?",
    "Describe the structure of atoms",
    "What causes aurora borealis?",
    "How does nuclear fusion work?",
    "What is genetic mutation?",
    "Explain plate tectonics",
    "How do black holes form?",
    "What is RNA?",
    "Describe cellular respiration",
    "What causes hurricanes?",
    "How does MRI imaging work?",
    "What is string theory?",
    "Explain the Big Bang theory",
    "How do antibiotics work?",
    "What is gene therapy?",
    "Describe ocean currents",
    "What causes lightning?",
    "How does GPS work?",
    "What is dark energy?",
    "Explain the Higgs boson",
    "How do batteries store energy?",
    "What is CRISPR?",
    "Describe the ozone layer",
    "What causes tsunamis?",
    "How does radar work?",
    "What is entropy?",
    "Explain superconductivity",
]


async def send_request(
    session: aiohttp.ClientSession, url: str, query: str, k: int, return_fulltext: bool, rerank: bool = True
) -> Dict[str, Any]:
    """Send a single search request and measure timing."""
    payload = {"query": query, "k": k, "return_fulltext": return_fulltext, "rerank": rerank}

    start_time = time.perf_counter()

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            data = await response.json()
            end_time = time.perf_counter()

            return {
                "success": response.status == 200,
                "status_code": response.status,
                "latency": end_time - start_time,
                "query": query,
                "k": k,
                "return_fulltext": return_fulltext,
                "num_results": (
                    len(data.get("results", [])) if response.status == 200 else 0
                ),
                "error": None,
            }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "status_code": None,
            "latency": end_time - start_time,
            "query": query,
            "k": k,
            "return_fulltext": return_fulltext,
            "num_results": 0,
            "error": str(e),
        }


async def warmup(url: str, num_warmup: int = 10):
    """Warmup phase to prepare caches."""
    print(f"\n{'='*80}")
    print(f"WARMUP PHASE ({num_warmup} requests)")
    print(f"{'='*80}")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_warmup):
            query = random.choice(TEST_QUERIES)
            k = random.choice([1, 5, 10])
            tasks.append(send_request(session, url, query, k, False))

        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks), total=num_warmup, desc="Warming up"
        ):
            result = await coro
            results.append(result)

    success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
    avg_latency = np.mean([r["latency"] for r in results])
    print(
        f"✓ Warmup complete: {success_rate:.1f}% success, avg latency: {avg_latency:.3f}s"
    )


async def run_load_test(
    url: str,
    num_requests: int,
    concurrency: int,
    k_values: List[int],
    return_fulltext: bool,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    """Run load test with specified concurrency."""

    # Generate request parameters
    requests_params = []
    for i in range(num_requests):
        query = random.choice(TEST_QUERIES)
        k = random.choice(k_values)
        requests_params.append((query, k))

    results = []

    async with aiohttp.ClientSession() as session:
        # Process in batches of 'concurrency'
        for i in range(0, num_requests, concurrency):
            batch = requests_params[i : i + concurrency]
            tasks = [
                send_request(session, url, query, k, return_fulltext, rerank)
                for query, k in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance metrics from results."""
    latencies = [r["latency"] for r in results]
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    total_time = sum(latencies)

    metrics = {
        "total_requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "success_rate": len(successes) / len(results) * 100 if results else 0,
        "total_time": total_time,
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "latency": {
            "min": np.min(latencies) if latencies else 0,
            "max": np.max(latencies) if latencies else 0,
            "mean": np.mean(latencies) if latencies else 0,
            "median": np.median(latencies) if latencies else 0,
            "p95": np.percentile(latencies, 95) if latencies else 0,
            "p99": np.percentile(latencies, 99) if latencies else 0,
            "std": np.std(latencies) if latencies else 0,
        },
    }

    return metrics


def print_metrics(metrics: Dict[str, Any], test_name: str):
    """Print metrics in a formatted way."""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")
    print(f"Total Requests:      {metrics['total_requests']}")
    print(
        f"Successful:          {metrics['successful_requests']} ({metrics['success_rate']:.2f}%)"
    )
    print(f"Failed:              {metrics['failed_requests']}")
    print(f"Total Time:          {metrics['total_time']:.2f}s")
    print(f"Throughput:          {metrics['throughput']:.2f} req/s")
    print(f"\nLatency Statistics:")
    print(f"  Min:               {metrics['latency']['min']*1000:.2f}ms")
    print(f"  Max:               {metrics['latency']['max']*1000:.2f}ms")
    print(f"  Mean:              {metrics['latency']['mean']*1000:.2f}ms")
    print(f"  Median (p50):      {metrics['latency']['median']*1000:.2f}ms")
    print(f"  p95:               {metrics['latency']['p95']*1000:.2f}ms")
    print(f"  p99:               {metrics['latency']['p99']*1000:.2f}ms")
    print(f"  Std Dev:           {metrics['latency']['std']*1000:.2f}ms")
    print(f"{'='*80}")


def plot_results(all_results: Dict[str, Any], output_dir: Path):
    """Generate performance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Latency comparison across concurrency levels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Without fulltext
    concurrency_levels = sorted(all_results["without_fulltext"].keys())
    metrics_without = [
        all_results["without_fulltext"][c]["metrics"] for c in concurrency_levels
    ]

    # Plot latency percentiles
    ax = axes[0, 0]
    ax.plot(
        concurrency_levels,
        [m["latency"]["mean"] * 1000 for m in metrics_without],
        "o-",
        label="Mean",
        linewidth=2,
    )
    ax.plot(
        concurrency_levels,
        [m["latency"]["median"] * 1000 for m in metrics_without],
        "s-",
        label="Median (p50)",
        linewidth=2,
    )
    ax.plot(
        concurrency_levels,
        [m["latency"]["p95"] * 1000 for m in metrics_without],
        "^-",
        label="p95",
        linewidth=2,
    )
    ax.plot(
        concurrency_levels,
        [m["latency"]["p99"] * 1000 for m in metrics_without],
        "v-",
        label="p99",
        linewidth=2,
    )
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        "Latency vs Concurrency (without fulltext)", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot throughput
    ax = axes[0, 1]
    ax.plot(
        concurrency_levels,
        [m["throughput"] for m in metrics_without],
        "o-",
        linewidth=2,
        color="green",
    )
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Throughput (req/s)", fontsize=12)
    ax.set_title(
        "Throughput vs Concurrency (without fulltext)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # With fulltext
    if all_results["with_fulltext"]:
        metrics_with = [
            all_results["with_fulltext"][c]["metrics"] for c in concurrency_levels
        ]

        # Plot latency percentiles
        ax = axes[1, 0]
        ax.plot(
            concurrency_levels,
            [m["latency"]["mean"] * 1000 for m in metrics_with],
            "o-",
            label="Mean",
            linewidth=2,
        )
        ax.plot(
            concurrency_levels,
            [m["latency"]["median"] * 1000 for m in metrics_with],
            "s-",
            label="Median (p50)",
            linewidth=2,
        )
        ax.plot(
            concurrency_levels,
            [m["latency"]["p95"] * 1000 for m in metrics_with],
            "^-",
            label="p95",
            linewidth=2,
        )
        ax.plot(
            concurrency_levels,
            [m["latency"]["p99"] * 1000 for m in metrics_with],
            "v-",
            label="p99",
            linewidth=2,
        )
        ax.set_xlabel("Concurrency Level", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title(
            "Latency vs Concurrency (with fulltext)", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot throughput
        ax = axes[1, 1]
        ax.plot(
            concurrency_levels,
            [m["throughput"] for m in metrics_with],
            "o-",
            linewidth=2,
            color="green",
        )
        ax.set_xlabel("Concurrency Level", fontsize=12)
        ax.set_ylabel("Throughput (req/s)", fontsize=12)
        ax.set_title(
            "Throughput vs Concurrency (with fulltext)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "load_test_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_file}")

    # Plot 2: Latency distribution histogram for each concurrency level
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, conc in enumerate(concurrency_levels):
        if idx >= 6:
            break
        ax = axes[idx]
        latencies = [
            r["latency"] * 1000
            for r in all_results["without_fulltext"][conc]["results"]
        ]
        ax.hist(latencies, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(
            np.mean(latencies),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(latencies):.1f}ms",
        )
        ax.axvline(
            np.median(latencies),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(latencies):.1f}ms",
        )
        ax.set_xlabel("Latency (ms)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Concurrency = {conc}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_file = output_dir / "latency_distributions.png"
    plt.savefig(hist_file, dpi=150, bbox_inches="tight")
    print(f"✓ Histogram saved to: {hist_file}")


async def main():
    parser = argparse.ArgumentParser(description="Load test FAISS search API")

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/search",
        help="API endpoint URL",
    )

    parser.add_argument(
        "--num_requests",
        type=int,
        default=1000,
        help="Total number of requests to send",
    )

    parser.add_argument(
        "--concurrency_levels",
        type=str,
        default="1,5,10,20,30,50",
        help="Comma-separated concurrency levels to test",
    )

    parser.add_argument(
        "--k_values",
        type=str,
        default="1,5,10",
        help="Comma-separated k values to test",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="load_test_results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup requests"
    )

    parser.add_argument(
        "--skip_fulltext",
        action="store_true",
        help="Skip testing with return_fulltext=true",
    )

    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]
    output_dir = Path(args.output_dir)

    print(f"{'='*80}")
    print(f"LOAD TEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"URL:                 {args.url}")
    print(f"Total Requests:      {args.num_requests}")
    print(f"Concurrency Levels:  {concurrency_levels}")
    print(f"K Values:            {k_values}")
    print(f"Test with fulltext:  {not args.skip_fulltext}")
    print(f"Warmup Requests:     {args.warmup}")
    print(f"Output Directory:    {output_dir}")
    print(f"{'='*80}")

    # Warmup
    if args.warmup > 0:
        await warmup(args.url, args.warmup)

    # Store all results
    all_results = {
        "without_fulltext": {},
        "with_fulltext": {},
        "without_fulltext_no_rerank": {},
        "with_fulltext_no_rerank": {},
        "config": {
            "url": args.url,
            "num_requests": args.num_requests,
            "concurrency_levels": concurrency_levels,
            "k_values": k_values,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Test without fulltext (WITH reranking)
    print(f"\n{'='*80}")
    print("TESTING WITHOUT FULLTEXT (WITH RERANKING)")
    print(f"{'='*80}")

    for concurrency in concurrency_levels:
        print(f"\n>>> Running with concurrency = {concurrency}")
        start_time = time.perf_counter()

        results = await run_load_test(
            args.url, args.num_requests, concurrency, k_values, return_fulltext=False, rerank=True
        )

        end_time = time.perf_counter()
        metrics = calculate_metrics(results)

        all_results["without_fulltext"][concurrency] = {
            "results": results,
            "metrics": metrics,
            "wall_time": end_time - start_time,
        }

        print(f"✓ Completed in {end_time - start_time:.2f}s")
        print(f"  Success rate: {metrics['success_rate']:.2f}%")
        print(f"  Throughput: {metrics['throughput']:.2f} req/s")
        print(f"  Mean latency: {metrics['latency']['mean']*1000:.2f}ms")
        print(f"  p95 latency: {metrics['latency']['p95']*1000:.2f}ms")

    # Test without fulltext (NO reranking)
    print(f"\n{'='*80}")
    print("TESTING WITHOUT FULLTEXT (NO RERANKING)")
    print(f"{'='*80}")

    for concurrency in concurrency_levels:
        print(f"\n>>> Running with concurrency = {concurrency}")
        start_time = time.perf_counter()

        results = await run_load_test(
            args.url, args.num_requests, concurrency, k_values, return_fulltext=False, rerank=False
        )

        end_time = time.perf_counter()
        metrics = calculate_metrics(results)

        all_results["without_fulltext_no_rerank"][concurrency] = {
            "results": results,
            "metrics": metrics,
            "wall_time": end_time - start_time,
        }

        print(f"✓ Completed in {end_time - start_time:.2f}s")
        print(f"  Success rate: {metrics['success_rate']:.2f}%")
        print(f"  Throughput: {metrics['throughput']:.2f} req/s")
        print(f"  Mean latency: {metrics['latency']['mean']*1000:.2f}ms")
        print(f"  p95 latency: {metrics['latency']['p95']*1000:.2f}ms")

    # Test with fulltext
    if not args.skip_fulltext:
        print(f"\n{'='*80}")
        print("TESTING WITH FULLTEXT (WITH RERANKING)")
        print(f"{'='*80}")

        for concurrency in concurrency_levels:
            print(f"\n>>> Running with concurrency = {concurrency}")
            start_time = time.perf_counter()

            results = await run_load_test(
                args.url, args.num_requests, concurrency, k_values, return_fulltext=True, rerank=True
            )

            end_time = time.perf_counter()
            metrics = calculate_metrics(results)

            all_results["with_fulltext"][concurrency] = {
                "results": results,
                "metrics": metrics,
                "wall_time": end_time - start_time,
            }

            print(f"✓ Completed in {end_time - start_time:.2f}s")
            print(f"  Success rate: {metrics['success_rate']:.2f}%")
            print(f"  Throughput: {metrics['throughput']:.2f} req/s")
            print(f"  Mean latency: {metrics['latency']['mean']*1000:.2f}ms")
            print(f"  p95 latency: {metrics['latency']['p95']*1000:.2f}ms")

        print(f"\n{'='*80}")
        print("TESTING WITH FULLTEXT (NO RERANKING)")
        print(f"{'='*80}")

        for concurrency in concurrency_levels:
            print(f"\n>>> Running with concurrency = {concurrency}")
            start_time = time.perf_counter()

            results = await run_load_test(
                args.url, args.num_requests, concurrency, k_values, return_fulltext=True, rerank=False
            )

            end_time = time.perf_counter()
            metrics = calculate_metrics(results)

            all_results["with_fulltext_no_rerank"][concurrency] = {
                "results": results,
                "metrics": metrics,
                "wall_time": end_time - start_time,
            }

            print(f"✓ Completed in {end_time - start_time:.2f}s")
            print(f"  Success rate: {metrics['success_rate']:.2f}%")
            print(f"  Throughput: {metrics['throughput']:.2f} req/s")
            print(f"  Mean latency: {metrics['latency']['mean']*1000:.2f}ms")
            print(f"  p95 latency: {metrics['latency']['p95']*1000:.2f}ms")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for concurrency in concurrency_levels:
        print_metrics(
            all_results["without_fulltext"][concurrency]["metrics"],
            f"WITHOUT FULLTEXT (WITH RERANKING) - Concurrency: {concurrency}",
        )

    for concurrency in concurrency_levels:
        print_metrics(
            all_results["without_fulltext_no_rerank"][concurrency]["metrics"],
            f"WITHOUT FULLTEXT (NO RERANKING) - Concurrency: {concurrency}",
        )

    if not args.skip_fulltext:
        for concurrency in concurrency_levels:
            print_metrics(
                all_results["with_fulltext"][concurrency]["metrics"],
                f"WITH FULLTEXT (WITH RERANKING) - Concurrency: {concurrency}",
            )

        for concurrency in concurrency_levels:
            print_metrics(
                all_results["with_fulltext_no_rerank"][concurrency]["metrics"],
                f"WITH FULLTEXT (NO RERANKING) - Concurrency: {concurrency}",
            )

    # Save results to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare JSON-serializable results (remove raw results, keep only metrics)
    json_results = {
        "config": all_results["config"],
        "without_fulltext": {
            str(k): {"metrics": v["metrics"], "wall_time": v["wall_time"]}
            for k, v in all_results["without_fulltext"].items()
        },
        "without_fulltext_no_rerank": {
            str(k): {"metrics": v["metrics"], "wall_time": v["wall_time"]}
            for k, v in all_results["without_fulltext_no_rerank"].items()
        },
        "with_fulltext": (
            {
                str(k): {"metrics": v["metrics"], "wall_time": v["wall_time"]}
                for k, v in all_results["with_fulltext"].items()
            }
            if not args.skip_fulltext
            else {}
        ),
        "with_fulltext_no_rerank": (
            {
                str(k): {"metrics": v["metrics"], "wall_time": v["wall_time"]}
                for k, v in all_results["with_fulltext_no_rerank"].items()
            }
            if not args.skip_fulltext
            else {}
        ),
    }

    results_file = output_dir / f"load_test_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Generate plots
    plot_results(all_results, output_dir)

    print(f"\n{'='*80}")
    print("LOAD TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
