import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_single_file(args):
    """Process a single JSONL file and return its mappings."""
    filepath, file_shard_id = args

    file_ids = []
    positions = []

    with open(filepath, "rb") as f:
        position = 0

        while True:
            current_position = position
            line = f.readline()

            if not line:
                break

            # Just check if it's valid (has newline and non-empty)
            if line.strip():
                file_ids.append(file_shard_id)
                positions.append(current_position)

            position = f.tell()

    return np.array(file_ids, dtype=np.int32), np.array(positions, dtype=np.int64)


def create_document_mappings(document_dir, output_dir, n_workers=None):
    """
    Creates doc_id_to_file_id and doc_pos_id_array from document JSONL files.

    Args:
        document_dir: Directory containing document JSONL files
        output_dir: Directory to save the numpy arrays
        n_workers: Number of parallel workers (defaults to CPU count)
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Get all JSONL files sorted by name for consistency
    doc_files = sorted(Path(document_dir).glob("*.jsonl"))

    # Store filenames (without path)
    doc_filenames = [f.name for f in doc_files]

    print(f"Processing {len(doc_files)} files using {n_workers} workers...")

    # Prepare arguments for parallel processing
    tasks = [
        (filepath, file_shard_id) for file_shard_id, filepath in enumerate(doc_files)
    ]

    # Process files in parallel
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_file, tasks),
                total=len(tasks),
                desc="Processing JSONL files",
            )
        )

    # Concatenate all results
    all_file_ids = []
    all_positions = []

    for file_ids, positions in results:
        all_file_ids.append(file_ids)
        all_positions.append(positions)

    doc_id_to_file_id = np.concatenate(all_file_ids)
    doc_pos_id_array = np.concatenate(all_positions)
    doc_filenames_array = np.array(doc_filenames, dtype=object)

    # Save arrays
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "doc_filenames.npy", doc_filenames_array)
    np.save(output_path / "doc_id_to_file_id.npy", doc_id_to_file_id)
    np.save(output_path / "doc_pos_id_array.npy", doc_pos_id_array)

    print(f"\n✓ Created mappings for {len(doc_id_to_file_id)} documents")
    print(f"✓ Saved to {output_dir}/")

    return doc_filenames_array, doc_id_to_file_id, doc_pos_id_array


def test_document_retrieval(
    doc_dir, doc_filenames, doc_id_to_file_id, doc_pos_id_array, test_doc_id=0
):
    """
    Test that we can retrieve a document correctly using the mappings.
    """
    # Get file and position for this doc_id
    file_shard_id = doc_id_to_file_id[test_doc_id]
    filename = doc_filenames[file_shard_id]
    position = doc_pos_id_array[test_doc_id]

    # Retrieve document
    filepath = Path(doc_dir) / filename
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(position)
        line = f.readline()
        doc = json.loads(line)

    print(f"\nTest retrieval for doc_id={test_doc_id}:")
    print(f"  File: {filename}")
    print(f"  Position: {position}")
    print(f"  Document: {doc}")

    return doc


# Usage example
if __name__ == "__main__":
    # Directory containing your original document JSONL files
    document_dir = "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_full/"

    # Where to save the numpy arrays
    output_dir = "/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_full_output/index/"

    doc_filenames, doc_id_to_file_id, doc_pos_id_array = create_document_mappings(
        document_dir, output_dir
    )

    # Verify the results
    print("\nVerification:")
    print(f"doc_filenames shape: {doc_filenames.shape}")
    print(f"doc_filenames: {doc_filenames}")
    print(f"\ndoc_id_to_file_id shape: {doc_id_to_file_id.shape}")
    print(f"First 10 mappings: {doc_id_to_file_id[:10]}")
    print(f"\ndoc_pos_id_array shape: {doc_pos_id_array.shape}")
    print(f"First 10 positions: {doc_pos_id_array[:10]}")

    # Test it
    doc_filenames = np.load(
        os.path.join(output_dir, "doc_filenames.npy"), allow_pickle=True
    )
    doc_id_to_file_id = np.load(os.path.join(output_dir, "doc_id_to_file_id.npy"))
    doc_pos_id_array = np.load(os.path.join(output_dir, "doc_pos_id_array.npy"))

    test_document_retrieval(
        document_dir, doc_filenames, doc_id_to_file_id, doc_pos_id_array, test_doc_id=0
    )
    test_document_retrieval(
        document_dir, doc_filenames, doc_id_to_file_id, doc_pos_id_array, test_doc_id=1
    )
