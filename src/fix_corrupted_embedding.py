#!/usr/bin/env python3
"""
Regenerate embeddings for a single corrupted passage file.
Uses 8 GPUs with DataParallel.
"""

import json
import torch
import pickle
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


class PassageDataset(Dataset):
    """Dataset that reads passages from JSONL file in order."""

    def __init__(self, jsonl_path, tokenizer, max_length=128):
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


def main(args):
    print("="*80)
    print("REGENERATE CORRUPTED EMBEDDING")
    print("="*80)

    # Setup paths
    passage_file = Path(args.passage_file)

    # Determine output path
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Auto-generate output path
        embeddings_dir = Path(args.embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_filename = passage_file.name.replace("_passages.jsonl", "_embeddings.pkl")
        output_file = embeddings_dir / embedding_filename

    print(f"\nInput: {passage_file}")
    print(f"Output: {output_file}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"GPUs available: {torch.cuda.device_count()}")

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

    # Create dataset
    print(f"\nLoading passages from {passage_file.name}...")
    dataset = PassageDataset(passage_file, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Total passages: {len(dataset):,}")
    print(f"Batches: {len(dataloader):,}")

    # Process batches
    print(f"\nGenerating embeddings...")
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            inputs = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"\nEmbeddings shape: {final_embeddings.shape}")

    # Save to pickle
    print(f"Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(final_embeddings, f)

    print(f"\n{'='*80}")
    print("✅ EMBEDDING REGENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Saved {len(final_embeddings):,} embeddings to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate embeddings for corrupted passage file")

    parser.add_argument(
        "--passage_file",
        type=str,
        required=True,
        help="Path to the passage JSONL file"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output pickle file path (auto-generated if not specified)"
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/embeddings",
        help="Directory to save embeddings (used if output_file not specified)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/contriever",
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size for embedding generation"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of DataLoader workers"
    )

    args = parser.parse_args()

    main(args)
