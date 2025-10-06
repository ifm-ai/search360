#!/usr/bin/env python3
"""
Create FAISS IVFPQ index from passage embeddings.
Maintains passage_id ordering for retrieval.
"""

import os
import pickle
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import faiss


def load_all_embeddings(embeddings_dir, passage_filenames):
    """
    Load embeddings in the exact order specified by passage_filenames.
    This maintains passage_id ordering.
    """
    print("\n" + "=" * 80)
    print("LOADING EMBEDDINGS IN ORDER")
    print("=" * 80)

    all_embeddings = []
    total_passages = 0

    for idx, passage_filename in enumerate(
        tqdm(passage_filenames, desc="Loading embeddings")
    ):
        # Get corresponding embedding file
        embedding_filename = passage_filename.replace(
            "_passages.jsonl", "_embeddings.pkl"
        )
        embedding_path = embeddings_dir / embedding_filename

        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

        # Load embeddings
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)

        all_embeddings.append(embeddings)
        total_passages += len(embeddings)

        if (idx + 1) % 100 == 0:
            print(
                f"  Loaded {idx+1}/{len(passage_filenames)} files, {total_passages:,} passages so far"
            )

    # Concatenate all embeddings
    print("\nConcatenating all embeddings...")
    all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    print(f"✓ Loaded {len(all_embeddings):,} embeddings")
    print(f"✓ Shape: {all_embeddings.shape}")

    return all_embeddings


def load_and_sample_file(args):
    """Load embeddings from one file and sample from it."""
    embedding_path, num_samples = args

    with open(embedding_path, "rb") as f:
        embeddings = pickle.load(f)

    # Sample from this file
    file_size = len(embeddings)
    actual_samples = min(num_samples, file_size)

    indices = np.random.choice(file_size, size=actual_samples, replace=False)
    sampled = embeddings[indices]

    return sampled


def sample_embeddings_for_training(
    embeddings_dir, passage_filenames, sample_size, sample_path=None, n_workers=32
):
    """
    Sample embeddings for index training using parallel loading.
    """
    print("\n" + "=" * 80)
    print(f"SAMPLING {sample_size:,} EMBEDDINGS FOR TRAINING")
    print("=" * 80)

    if sample_path and os.path.exists(sample_path):
        print(f"Loading pre-sampled embeddings from {sample_path}")
        with open(sample_path, "rb") as f:
            sampled_embeddings = pickle.load(f)
        print(f"✓ Loaded {len(sampled_embeddings):,} sampled embeddings")
        return sampled_embeddings

    per_file_sample = max(1, sample_size // len(passage_filenames))

    print(f"Sampling ~{per_file_sample:,} embeddings per file from {len(passage_filenames)} files")
    print(f"Using {n_workers} parallel workers for loading")

    # Prepare tasks
    tasks = []
    for passage_filename in passage_filenames:
        embedding_filename = passage_filename.replace("_passages.jsonl", "_embeddings.pkl")
        embedding_path = embeddings_dir / embedding_filename
        tasks.append((embedding_path, per_file_sample))

    # Load and sample in parallel
    from multiprocessing import Pool
    with Pool(n_workers) as pool:
        sampled_embeddings = list(tqdm(
            pool.imap(load_and_sample_file, tasks),
            total=len(tasks),
            desc="Sampling files in parallel"
        ))

    sampled_embeddings = np.concatenate(sampled_embeddings, axis=0).astype(np.float32)

    # If we have more than needed, subsample
    if len(sampled_embeddings) > sample_size:
        indices = np.random.choice(
            len(sampled_embeddings), size=sample_size, replace=False
        )
        sampled_embeddings = sampled_embeddings[indices]

    print(f"✓ Sampled {len(sampled_embeddings):,} embeddings for training")

    # Save for future use
    if sample_path:
        print(f"Saving sampled embeddings to {sample_path}")
        with open(sample_path, "wb") as f:
            pickle.dump(sampled_embeddings, f)

    return sampled_embeddings


def train_index(
    sampled_embeddings,
    dimension,
    ncentroids,
    n_subquantizers,
    code_size,
    nprobe,
    use_gpu=True,
):
    """
    Train IVFPQ index on sampled embeddings.
    """
    print("\n" + "=" * 80)
    print("TRAINING IVFPQ INDEX")
    print("=" * 80)
    print(f"  Dimension: {dimension}")
    print(f"  Centroids: {ncentroids}")
    print(f"  Subquantizers: {n_subquantizers}")
    print(f"  Code size: {code_size}")
    print(f"  Probe: {nprobe}")
    print(f"  Training samples: {len(sampled_embeddings):,}")

    # Create index
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        ncentroids,
        n_subquantizers,
        code_size,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = nprobe

    # Set random seed for reproducibility
    np.random.seed(1234)
    index.cp.seed = 1234
    index.pq.cp.seed = 1234

    print("\nTraining index...")
    start_time = time.time()

    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Using GPU for training (GPUs available: {faiss.get_num_gpus()})")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True

        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        gpu_index.verbose = False

        # Train on GPU
        gpu_index.train(sampled_embeddings)

        # Move back to CPU
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        print("Training on CPU")
        index.train(sampled_embeddings)

    training_time = time.time() - start_time
    print(
        f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)"
    )

    assert index.is_trained
    print(f"✓ Index is trained: {index.is_trained}")
    print(f"✓ Index total: {index.ntotal}")

    return index


def add_embeddings_to_index(
    index, embeddings_dir, passage_filenames, batch_size=1000000
):
    """
    Add embeddings to trained index in order.
    CRITICAL: This must add embeddings in the exact order of passage_filenames
    to maintain passage_id == faiss_id invariant.
    """
    print("\n" + "=" * 80)
    print("ADDING EMBEDDINGS TO INDEX")
    print("=" * 80)
    print(f"  Batch size: {batch_size:,}")

    assert index.is_trained
    assert index.ntotal == 0, "Index should be empty before adding embeddings"

    total_added = 0
    start_time = time.time()

    # Add embeddings file by file in exact order
    for idx, passage_filename in enumerate(passage_filenames):
        embedding_filename = passage_filename.replace(
            "_passages.jsonl", "_embeddings.pkl"
        )
        embedding_path = embeddings_dir / embedding_filename

        # Load embeddings
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)

        embeddings = embeddings.astype(np.float32)

        # Add in batches if file is large
        num_embeddings = len(embeddings)
        for i in range(0, num_embeddings, batch_size):
            batch = embeddings[i : i + batch_size]
            index.add(batch)
            total_added += len(batch)

        elapsed = time.time() - start_time
        print(
            f"  [{idx+1}/{len(passage_filenames)}] Added {embedding_filename}: "
            f"{total_added:,} total passages ({elapsed/60:.2f} min)"
        )

    print(f"\n✓ Added {total_added:,} embeddings to index")
    print(f"✓ Index ntotal: {index.ntotal}")
    print(f"✓ Time taken: {(time.time() - start_time)/60:.2f} minutes")

    assert index.ntotal == total_added, f"Mismatch: {index.ntotal} != {total_added}"

    return index


def verify_index(index, passage_id_to_file_id):
    """
    Verify that index size matches mapping arrays.
    """
    print("\n" + "=" * 80)
    print("VERIFYING INDEX")
    print("=" * 80)

    print(f"  FAISS index ntotal: {index.ntotal:,}")
    print(f"  passage_id_to_file_id length: {len(passage_id_to_file_id):,}")

    if index.ntotal == len(passage_id_to_file_id):
        print("✓ Index size matches mapping arrays!")
    else:
        print(f"✗ MISMATCH: {index.ntotal} != {len(passage_id_to_file_id)}")
        raise ValueError("Index size does not match mapping arrays")


def main(args):
    print("=" * 80)
    print("FAISS IVFPQ INDEX CREATION")
    print("=" * 80)

    # Setup paths
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir) / "index_faiss"
    output_dir.mkdir(parents=True, exist_ok=True)

    trained_index_path = output_dir / "trained_index.faiss"
    final_index_path = output_dir / "final_index.faiss"
    sample_path = output_dir / "training_samples.pkl" if args.save_samples else None

    # Load passage filenames to maintain order
    mappings_dir = Path(args.output_dir) / "mappings_passages"
    passage_filenames = np.load(
        mappings_dir / "passage_filenames.npy", allow_pickle=True
    )
    passage_id_to_file_id = np.load(mappings_dir / "passage_id_to_file_id.npy")

    print(f"\nConfiguration:")
    print(f"  Embeddings dir: {embeddings_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Passage files: {len(passage_filenames)}")
    print(f"  Total passages: {len(passage_id_to_file_id):,}")
    print(f"  Dimension: {args.dimension}")
    print(f"  Centroids: {args.ncentroids}")
    print(f"  Probe: {args.nprobe}")
    print(f"  Subquantizers: {args.n_subquantizers}")
    print(f"  Code size: {args.code_size}")
    print(f"  Training samples: {args.sample_size:,}")

    # Step 1: Sample embeddings for training
    if not trained_index_path.exists() or args.retrain:
        sampled_embeddings = sample_embeddings_for_training(
            embeddings_dir, passage_filenames, args.sample_size, sample_path
        )

        # Step 2: Train index
        index = train_index(
            sampled_embeddings,
            args.dimension,
            args.ncentroids,
            args.n_subquantizers,
            args.code_size,
            args.nprobe,
            use_gpu=args.use_gpu,
        )

        # Save trained index
        print(f"\nSaving trained index to {trained_index_path}")
        faiss.write_index(index, str(trained_index_path))
        print("✓ Trained index saved")

        del sampled_embeddings  # Free memory
    else:
        print(f"\nLoading existing trained index from {trained_index_path}")
        index = faiss.read_index(str(trained_index_path))
        print("✓ Trained index loaded")

    # Step 3: Add all embeddings in order
    if not final_index_path.exists() or args.rebuild:
        # Reset index if it has data
        if index.ntotal > 0:
            print("\nResetting index (removing existing data)...")
            index.reset()

        index = add_embeddings_to_index(
            index, embeddings_dir, passage_filenames, batch_size=args.batch_size
        )

        # Step 4: Save final index
        print(f"\nSaving final index to {final_index_path}")
        faiss.write_index(index, str(final_index_path))
        print("✓ Final index saved")
    else:
        print(f"\nFinal index already exists at {final_index_path}")
        print("Use --rebuild to rebuild the index")

    # Step 5: Verify
    verify_index(index, passage_id_to_file_id)

    print("\n" + "=" * 80)
    print("✅ INDEX CREATION COMPLETE")
    print("=" * 80)
    print(f"Final index: {final_index_path}")
    print(f"Index contains {index.ntotal:,} passages")
    print(f"\nTo use this index, load:")
    print(f"  - Index: {final_index_path}")
    print(f"  - Passage mappings: {mappings_dir}/passage_*.npy")
    print(f"  - Document mappings: {args.output_dir}/index/doc_*.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create FAISS IVFPQ index from embeddings"
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/embeddings",
        help="Directory containing embedding pickle files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs",
        help="Output directory (index will be saved to output_dir/index_faiss/)",
    )

    parser.add_argument(
        "--dimension", type=int, default=768, help="Embedding dimension"
    )

    parser.add_argument(
        "--ncentroids", type=int, default=4096, help="Number of centroids for IVFPQ"
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=2048,
        help="Number of clusters to probe during search",
    )

    parser.add_argument(
        "--n_subquantizers", type=int, default=16, help="Number of subquantizers for PQ"
    )

    parser.add_argument("--code_size", type=int, default=8, help="Code size for PQ")

    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000000,
        help="Number of embeddings to sample for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000000,
        help="Batch size for adding embeddings to index",
    )

    parser.add_argument(
        "--use_gpu", action="store_true", default=True, help="Use GPU for training"
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain index even if trained index exists",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index even if final index exists",
    )

    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save sampled embeddings for future use",
    )

    args = parser.parse_args()

    main(args)
