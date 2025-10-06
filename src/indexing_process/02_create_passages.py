#!/usr/bin/env python3
"""
Creates passages from documents and builds mappings:
- Chunks each document into 128-token passages
- Merges last passage with second-to-last if < 64 tokens
- Creates passage_to_doc_id mapping (passage_id -> doc_id)
- Creates passage lookup arrays (passage_filenames, passage_id_to_file_id, passage_pos_id_array)


This step took around 7 hours to finish on 128 cores for total documents of 144996131
"""

import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import os
from pathlib import Path

# Set cache to current working directory
os.environ["HF_HOME"] = os.getcwd()

# Global tokenizer for worker processes
_worker_tokenizer = None


def init_worker(tokenizer_name):
    """Initialize worker with tokenizer (called once per worker)."""
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def chunk_text(text, tokenizer, chunk_size=128, min_last_chunk=64):
    """
    Chunk text into passages of chunk_size tokens.
    If the last chunk is < min_last_chunk tokens, merge it with the second-to-last chunk.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) == 0:
        return []

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(chunk_tokens)

    # Handle small last chunk
    if len(chunks) > 1 and len(chunks[-1]) < min_last_chunk:
        # Merge last chunk with second-to-last
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()

    # Convert token chunks back to text
    text_chunks = []
    for chunk_tokens in chunks:
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        text_chunks.append(chunk_text)

    return text_chunks


def process_document_file(args):
    """Process a single document JSONL file and create passages - streaming to disk."""
    global _worker_tokenizer

    (
        doc_filepath,
        start_doc_id,
        chunk_size,
        min_last_chunk,
        output_passages_path,
    ) = args

    # Use the pre-loaded tokenizer from worker init
    tokenizer = _worker_tokenizer

    passage_to_doc_ids = []
    passage_positions = []

    with open(doc_filepath, "r", encoding="utf-8") as f_in, open(
        output_passages_path, "w", encoding="utf-8"
    ) as f_out:

        doc_id = start_doc_id

        for line in f_in:
            if not line.strip():
                continue

            doc = json.loads(line)
            text = doc.get("text", "")

            # Chunk the document
            chunks = chunk_text(text, tokenizer, chunk_size, min_last_chunk)

            # Write passage objects directly to file
            for chunk in chunks:
                position = f_out.tell()
                f_out.write(json.dumps({"text": chunk, "doc_id": doc_id}) + "\n")

                passage_to_doc_ids.append(doc_id)
                passage_positions.append(position)

            doc_id += 1

    return np.array(passage_to_doc_ids, dtype=np.int64), np.array(
        passage_positions, dtype=np.int64
    )


def count_documents(doc_dir):
    """Count total documents across all files to assign doc_ids."""
    doc_files = sorted(Path(doc_dir).glob("*.jsonl"))

    doc_counts = []
    for filepath in tqdm(doc_files, desc="Counting documents"):
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        doc_counts.append(count)

    return doc_files, doc_counts


def create_passages_and_mappings(
    document_dir,
    output_dir,
    tokenizer_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
    chunk_size=128,
    min_last_chunk=64,
    n_workers=None,
):
    """
    Create passages from documents and build all necessary mappings.

    Args:
        document_dir: Directory containing original document JSONL files
        output_dir: Directory to save passage JSONL files and mappings
        tokenizer_name: HuggingFace tokenizer to use for chunking
        chunk_size: Number of tokens per passage
        min_last_chunk: Minimum tokens for last chunk (merge if smaller)
        n_workers: Number of parallel workers
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Create output directories
    passages_dir = Path(output_dir) / "passages"
    mappings_dir = Path(output_dir) / "mappings_passages"
    passages_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Count documents to assign doc_ids
    print("Step 1: Counting documents...")
    doc_files, doc_counts = count_documents(document_dir)

    # Calculate starting doc_id for each file
    start_doc_ids = [0]
    for count in doc_counts[:-1]:
        start_doc_ids.append(start_doc_ids[-1] + count)

    print(f"Total documents: {sum(doc_counts)}")
    print(f"Processing {len(doc_files)} document files using {n_workers} workers...")

    # Pre-create passage output file paths
    passage_filenames = []
    passage_output_paths = []

    for filepath in doc_files:
        doc_filename = filepath.stem
        passage_filename = f"{doc_filename}_passages.jsonl"
        passage_filepath = passages_dir / passage_filename

        passage_filenames.append(passage_filename)
        passage_output_paths.append(str(passage_filepath))

    # Prepare tasks with output paths (no tokenizer_name in args anymore)
    tasks = [
        (filepath, start_doc_id, chunk_size, min_last_chunk, output_path)
        for filepath, start_doc_id, output_path in zip(
            doc_files, start_doc_ids, passage_output_paths
        )
    ]

    # Process documents in parallel - passages written directly to disk
    print("\nStep 2: Creating passages and streaming to disk...")
    from functools import partial

    with Pool(n_workers, initializer=init_worker, initargs=(tokenizer_name,)) as pool:
        results = list(
            tqdm(
                pool.imap(process_document_file, tasks),
                total=len(tasks),
                desc="Creating passages",
            )
        )

    # Build mappings from results (only small arrays, no passage text)
    print("\nStep 3: Building mappings from results...")

    all_passage_to_doc_ids = []
    all_passage_file_ids = []
    all_passage_positions = []

    for file_id, (passage_to_doc_ids, passage_positions) in enumerate(
        tqdm(results, desc="Building mappings")
    ):
        if len(passage_to_doc_ids) == 0:
            continue

        # Track which file these passages belong to
        file_ids = np.full(len(passage_to_doc_ids), file_id, dtype=np.int32)

        all_passage_file_ids.append(file_ids)
        all_passage_positions.append(passage_positions)
        all_passage_to_doc_ids.append(passage_to_doc_ids)

    # Convert to numpy arrays
    print("\nStep 4: Creating numpy arrays...")
    passage_filenames_array = np.array(passage_filenames, dtype=object)
    passage_id_to_file_id = np.concatenate(all_passage_file_ids)
    passage_pos_id_array = np.concatenate(all_passage_positions)
    passage_to_doc_id = np.concatenate(all_passage_to_doc_ids)

    # Save all mappings
    np.save(mappings_dir / "passage_filenames.npy", passage_filenames_array)
    np.save(mappings_dir / "passage_id_to_file_id.npy", passage_id_to_file_id)
    np.save(mappings_dir / "passage_pos_id_array.npy", passage_pos_id_array)
    np.save(mappings_dir / "passage_to_doc_id.npy", passage_to_doc_id)

    print(f"\n✓ Created {len(passage_to_doc_id)} passages")
    print(f"✓ Passages saved to: {passages_dir}/")
    print(f"✓ Mappings saved to: {mappings_dir}/")
    print(f"\nMapping files created:")
    print(f"  - passage_filenames.npy: {len(passage_filenames_array)} files")
    print(f"  - passage_id_to_file_id.npy: {len(passage_id_to_file_id)} mappings")
    print(f"  - passage_pos_id_array.npy: {len(passage_pos_id_array)} positions")
    print(f"  - passage_to_doc_id.npy: {len(passage_to_doc_id)} passage→doc mappings")

    return {
        "passage_filenames": passage_filenames_array,
        "passage_id_to_file_id": passage_id_to_file_id,
        "passage_pos_id_array": passage_pos_id_array,
        "passage_to_doc_id": passage_to_doc_id,
    }


def test_passage_to_document_lookup(
    passages_dir,
    document_dir,
    passage_mappings,
    doc_filenames,
    doc_id_to_file_id,
    doc_pos_id_array,
    test_passage_id=0,
):
    """Test that we can retrieve both passage and original document, and time the lookup."""

    start_time = time.perf_counter()

    # Get passage
    psg_file_id = passage_mappings["passage_id_to_file_id"][test_passage_id]
    psg_filename = passage_mappings["passage_filenames"][psg_file_id]
    psg_position = passage_mappings["passage_pos_id_array"][test_passage_id]

    with open(Path(passages_dir) / psg_filename, "r") as f:
        f.seek(psg_position)
        passage = json.loads(f.readline())

    # Get original document
    doc_id = passage_mappings["passage_to_doc_id"][test_passage_id]
    doc_file_id = doc_id_to_file_id[doc_id]
    doc_filename = doc_filenames[doc_file_id]
    doc_position = doc_pos_id_array[doc_id]

    with open(Path(document_dir) / doc_filename, "r") as f:
        f.seek(doc_position)
        document = json.loads(f.readline())

    elapsed = time.perf_counter() - start_time

    print(f"\n=== Test: passage_id={test_passage_id} ===")
    print(f"Passage file: {psg_filename}")
    print(f"Passage: {passage['text'][:100]}...")
    print(f"\nOriginal doc_id: {doc_id}")
    print(f"Document file: {doc_filename}")
    print(f"Document: {document.get('text', '')[:100]}...")
    print(f"\nLookup time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    # Directories
    document_dir = "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_full"
    output_dir = "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_full_output"

    # Create passages and mappings
    passage_mappings = create_passages_and_mappings(
        document_dir=document_dir,
        output_dir=output_dir,
        tokenizer_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        chunk_size=128,
        min_last_chunk=64,
        n_workers=cpu_count(),
    )

    # Load document mappings for testing
    mappings_dir = Path(output_dir) / "index"
    doc_filenames = np.load(mappings_dir / "doc_filenames.npy", allow_pickle=True)
    doc_id_to_file_id = np.load(mappings_dir / "doc_id_to_file_id.npy")
    doc_pos_id_array = np.load(mappings_dir / "doc_pos_id_array.npy")

    # Test retrieval
    test_passage_to_document_lookup(
        passages_dir=Path(output_dir) / "passages",
        document_dir=document_dir,
        passage_mappings=passage_mappings,
        doc_filenames=doc_filenames,
        doc_id_to_file_id=doc_id_to_file_id,
        doc_pos_id_array=doc_pos_id_array,
        test_passage_id=0,
    )
