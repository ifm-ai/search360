#!/usr/bin/env python3
"""
Create embeddings for all passages using Contriever model.
Maintains strict passage_id ordering for FAISS indexing.
"""

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse


class PassageDataset(Dataset):
    """Dataset that reads passages from JSONL file in order."""

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load all passages in order
        self.passages = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    passage = json.loads(line)
                    self.passages.append(passage["text"])

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.passages[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def mean_pooling(token_embeddings, mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def embed_passage_file(
    passage_file,
    output_file,
    model,
    tokenizer,
    batch_size=1024,
    max_length=512,
    num_workers=8,
):
    """Create embeddings for a single passage file."""

    print(f"\nProcessing: {passage_file.name}")

    # Create dataset
    dataset = PassageDataset(passage_file, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: maintain order
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Total passages: {len(dataset):,}")
    print(f"  Batches: {len(dataloader):,}")

    # Process batches
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Embedding {passage_file.name}"):
            inputs = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"  Embeddings shape: {final_embeddings.shape}")

    # Save to pickle
    with open(output_file, "wb") as f:
        pickle.dump(final_embeddings, f)

    print(f"  ✓ Saved to: {output_file}")

    return len(final_embeddings)


def main(args):
    print("=" * 80)
    print("PASSAGE EMBEDDING CREATION")
    print("=" * 80)

    # Setup paths
    passages_dir = Path(args.passages_dir)
    embeddings_dir = Path(args.output_dir) / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings to get passage file order
    mappings_dir = Path(args.output_dir) / "mappings_passages"
    passage_filenames = np.load(
        mappings_dir / "passage_filenames.npy", allow_pickle=True
    )

    # Calculate file chunk for this job
    total_files = len(passage_filenames)
    if args.job_id is not None and args.total_jobs is not None:
        files_per_job = total_files // args.total_jobs
        remainder = total_files % args.total_jobs

        # Distribute remainder files to first jobs
        if args.job_id < remainder:
            start_idx = args.job_id * (files_per_job + 1)
            end_idx = start_idx + files_per_job + 1
        else:
            start_idx = args.job_id * files_per_job + remainder
            end_idx = start_idx + files_per_job

        # Get this job's file chunk
        passage_filenames = passage_filenames[start_idx:end_idx]

        print(f"\n{'='*80}")
        print(f"SLURM JOB CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Job ID: {args.job_id}")
        print(f"  Total jobs: {args.total_jobs}")
        print(f"  Total files: {total_files}")
        print(
            f"  This job processes files: [{start_idx}:{end_idx}] ({len(passage_filenames)} files)"
        )

    print(f"\nConfiguration:")
    print(f"  Passages dir: {passages_dir}")
    print(f"  Embeddings dir: {embeddings_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  GPUs available: {torch.cuda.device_count()}")
    print(f"  Files to process: {len(passage_filenames)}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    # Use all available GPUs with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to("cuda")
    model.eval()

    print("\n" + "=" * 80)
    print("PROCESSING PASSAGE FILES IN ORDER")
    print("=" * 80)

    # Process passage files in the EXACT order they appear in passage_filenames
    # This maintains passage_id ordering for FAISS
    total_passages = 0

    for idx, filename in enumerate(passage_filenames):
        passage_file = passages_dir / filename

        # Create corresponding embedding filename
        embedding_filename = filename.replace("_passages.jsonl", "_embeddings.pkl")
        output_file = embeddings_dir / embedding_filename

        # Skip if already exists (for resuming)
        if output_file.exists() and not args.overwrite:
            print(
                f"\n[{idx+1}/{len(passage_filenames)}] Skipping {filename} (already exists)"
            )
            continue

        print(f"\n[{idx+1}/{len(passage_filenames)}] Processing: {filename}")

        num_embeddings = embed_passage_file(
            passage_file=passage_file,
            output_file=output_file,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
        )

        total_passages += num_embeddings

    print("\n" + "=" * 80)
    print("✅ EMBEDDING CREATION COMPLETE")
    print("=" * 80)
    print(f"Total passages embedded: {total_passages:,}")
    print(f"Embeddings saved to: {embeddings_dir}")
    print(f"\nEmbedding files created: {len(list(embeddings_dir.glob('*.pkl')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create passage embeddings for FAISS indexing"
    )

    parser.add_argument(
        "--passages_dir",
        type=str,
        default="outputs/passages",
        help="Directory containing passage JSONL files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (embeddings will be saved to output_dir/embeddings/)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/contriever",
        help="HuggingFace model name for embeddings",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for embedding generation",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--num_workers", type=int, default=128, help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing embedding files"
    )

    parser.add_argument(
        "--job_id",
        type=int,
        default=None,
        help="Job ID for parallel processing (0-indexed). Use with --total_jobs",
    )

    parser.add_argument(
        "--total_jobs",
        type=int,
        default=None,
        help="Total number of parallel jobs. Use with --job_id",
    )

    args = parser.parse_args()

    main(args)
