#!/usr/bin/env python3
"""
Fast multi-core parquet to JSONL converter.
Processes all subdirectories in hq_parquets and converts parquet files to JSONL.
"""

import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pyarrow.parquet as pq
from tqdm import tqdm


def convert_parquet_to_jsonl(args):
    """Convert a single parquet file to JSONL."""
    parquet_path, output_path = args

    try:
        # Read parquet file
        table = pq.read_table(parquet_path)

        # Convert to list of dicts
        records = table.to_pylist()

        # Write to JSONL
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

        return f"✓ {output_path}"
    except Exception as e:
        return f"✗ {parquet_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to JSONL format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="hq_parquets",
        help="Input directory containing parquet files (default: hq_parquets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="documents_jsonl",
        help="Output directory for JSONL files (default: documents_jsonl)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Find all parquet files in subdirectories
    tasks = []
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            subdir_name = subdir.name
            for parquet_file in sorted(subdir.glob("*.parquet")):
                # Extract part number from filename (e.g., part_0000.parquet)
                part_name = parquet_file.stem  # e.g., part_0000

                # Create output filename: subdirname_partname.jsonl
                output_filename = f"{subdir_name}_{part_name}.jsonl"
                output_path = output_dir / output_filename

                tasks.append((str(parquet_file), str(output_path)))

    print(f"Found {len(tasks)} parquet files to convert")
    print(f"Using {cpu_count()} CPU cores")

    # Process in parallel with progress bar
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(convert_parquet_to_jsonl, tasks),
            total=len(tasks),
            desc="Converting parquet to JSONL"
        ))

    # Print results
    print("\nConversion complete!")
    print(f"Output directory: {output_dir}")

    # Count successes and failures
    successes = sum(1 for r in results if r.startswith("✓"))
    failures = sum(1 for r in results if r.startswith("✗"))
    print(f"Success: {successes}, Failures: {failures}")

    if failures > 0:
        print("\nFailed conversions:")
        for r in results:
            if r.startswith("✗"):
                print(r)


if __name__ == "__main__":
    main()
