# PretrainSeer

Search and inspect billions of documents in your pretraining corpus.

## Overview

This project provides:
- **Document processing pipeline**: Convert documents to passages with byte-offset indexing
- **Embedding generation**: Multi-GPU embedding using facebook/contriever
- **FAISS indexing**: IVFPQ index for efficient billion-scale search
- **FastAPI service**: REST API for passage and document retrieval
- **Load testing**: Comprehensive benchmarking tools

## Quick Start

### 1. Install Dependencies

```bash
conda env create -f environment.yml -p .conda
conda activate ./.conda
```

All commands below assume this environment is active (or prefix them with
`.conda/bin/python`).

### 2. Point the service at your data

The service needs a prebuilt FAISS index plus the passage/document files it was
built from. **These artifacts are not distributed with this repo** — build them
with the [pipeline below](#data-processing-pipeline), then point the server at
them with either CLI flags or environment variables:

| Setting | Flag | Environment variable |
|---|---|---|
| FAISS index file | `--index_path` | `INDEX_PATH` |
| Mappings directory | `--output_dir` | `OUTPUT_DIR` |
| Passage JSONL directory | `--passages_dir` | `PASSAGES_DIR` |
| Document JSONL directory | `--documents_dir` | `DOCUMENTS_DIR` |
| Embedding model | `--model_name` | `MODEL_NAME` |
| Clusters probed per query | `--nprobe` | `NPROBE` |
| FAISS on GPU | `--use_gpu` / `--no-use_gpu` | `USE_GPU` |
| Load cross-encoder re-ranker | `--load_reranker` / `--no-load_reranker` | `LOAD_RERANKER` |

CLI flags take precedence over environment variables, which take precedence over
the defaults in `api/server.py`.

### 3. Start the API Service

**Local (GPU, the default):**
```bash
python -m api.server --index_path /path/to/final_index.faiss --workers 8
```

**Local (CPU):**
```bash
python -m api.server --no-use_gpu --no-load_reranker \
  --index_path /path/to/final_index.faiss
```

For 8x H200 I recommend 16 workers.

**SLURM (Production):**
```bash
sbatch slurm_start_service.sh
# Check status: cat runtime/server_info_<job_id>.json
```

### 4. Test the Service

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 10,
    "return_fulltext": false
  }'
```

## Project Structure

```
pretrainseer/
├── src/indexing_process/                 # Data processing pipeline (run in order)
│   ├── 00_convert_parquets_to_jsonl.py   # Convert parquet to JSONL
│   ├── 01_map_documents.py               # Create document mappings
│   ├── 02_create_passages.py             # Chunk documents into passages
│   ├── 03_create_embeddings.py           # Generate embeddings (multi-GPU)
│   └── 04_create_faiss_index.py          # Build FAISS index
│
├── scripts/
│   └── slurm_create_embeddings.sh        # SLURM script for embeddings
│
├── api/                              # FastAPI service
│   ├── server.py                     # Main API server
│   ├── search.py                     # Search logic and index loading
│   ├── utils.py                      # Document ranking utilities
│   ├── benchmark.py                  # Single-query benchmark (10 queries)
│   └── load_test.py                  # Load testing (1000+ requests)
│
├── slurm_start_service.sh            # SLURM script to start API service
├── environment.yml                   # Conda environment export
├── API_README.md                     # Detailed API documentation
└── README.md                         # This file
```

## Key Files

### API Service
- **`api/server.py`**: FastAPI server with `/search`, `/get_document`, `/health`, `/stats` endpoints
- **`api/search.py`**: Core search functions (embedding, passage/document retrieval)
- **`API_README.md`**: Complete API documentation with examples

### Data Pipeline
- **`src/indexing_process/02_create_passages.py`**: Chunks documents into 128-token passages
- **`src/indexing_process/03_create_embeddings.py`**: Generates embeddings using DataParallel
- **`src/indexing_process/04_create_faiss_index.py`**: Creates the IVFPQ index

### Testing
- **`api/benchmark.py`**: Tests 10 queries, detailed timing breakdown
- **`api/load_test.py`**: Stress test with configurable concurrency (1-50 concurrent requests)

## System Requirements

**For API Service (16 workers):**
- 8 GPUs
- 600GB RAM
- 108GB VRAM per GPU

**For API Service (8 workers):**
- 8 GPUs
- 303GB RAM
- 54GB VRAM per GPU

## Data Processing Pipeline

Run the stages in order. Each stage writes into the output root that the next
stage reads, so use one consistent root throughout.

```bash
# 1. Convert parquet to JSONL
python src/indexing_process/00_convert_parquets_to_jsonl.py

# 2. Create document mappings
python src/indexing_process/01_map_documents.py

# 3. Create passages
python src/indexing_process/02_create_passages.py

# 4. Generate embeddings (multi-GPU; adapt the SLURM header to your cluster)
sbatch scripts/slurm_create_embeddings.sh

# 5. Build FAISS index
python src/indexing_process/04_create_faiss_index.py --use_gpu
```

> **Note:** stages 01 and 02 currently hardcode their input/output roots at the
> top of each file — edit those before running. `scripts/slurm_create_embeddings.sh`
> carries a site-specific SLURM header (partition, QoS, account) and an absolute
> interpreter path; adapt both to your cluster.

## API Endpoints

- **POST `/search`**: Search for passages (with optional document retrieval)
- **POST `/get_document`**: Get full document by ID
- **GET `/health`**: Health check
- **GET `/stats`**: Index statistics

See **[API_README.md](API_README.md)** for detailed endpoint documentation.

## Running Load Tests

```bash
# Test with default settings (1000 requests, concurrency 1-50)
python api/load_test.py --url http://localhost:8000/search

# Custom configuration
python api/load_test.py \
  --url http://your-server:8000/search \
  --num_requests 2000 \
  --concurrency_levels "1,10,25,50" \
  --k_values "5,10,20"
```

Generates:
- JSON results with latency metrics (min, max, mean, p50, p95, p99)
- Performance plots (latency vs concurrency, throughput)
- Latency distribution histograms

## Performance

**GPU (8 H200 GPUs, 16 workers):**
- Search: ~10-50ms per query
- Startup: ~30-60 seconds (loading index + model)
- Throughput: Up to 50+ req/s at concurrency=50

**Index Stats:**
- Total passages: 1.6B
- Total documents: 145M
- Index type: IVFPQ (4096 clusters, 16 subquantizers, 8-bit codes)

### Load Test Results (16 workers, with and without fulltext)

**Latency and Throughput:**

![Load Test Results](load_test_results_16_workers/load_test_results.png)

**Latency Distributions:**

![Latency Distributions](load_test_results_16_workers/latency_distributions.png)

Key metrics from load testing:
- **p95 latency**: ~0.5-1 sec (depending on concurrency)
- **p99 latency**: ~1-2 sec (depending on concurrency)
- **Throughput**: Scales well with concurrency (1→50 concurrent requests)
- **Success rate**: 100% across all concurrency levels

## SLURM Integration

> **Adapt these scripts before use.** `slurm_start_service.sh` and
> `scripts/slurm_create_embeddings.sh` carry a site-specific SLURM header
> (partition, QoS, account, reservation) and absolute interpreter paths from the
> original cluster — including a `projects/faster_index` path predating the
> rename. Edit both for your own site.

Start service and track status:
```bash
# Submit job
sbatch slurm_start_service.sh

# Monitor status
cat runtime/server_info_<job_id>.json

# Use in scripts
python your_script.py --job_id <job_id>
```

The server info file contains:
- Host and port
- Full URL
- Status (`loading` or `ready`)
- Timestamps

See **[API_README.md](API_README.md#running-with-slurm)** for programmatic access examples.

## Re-ranking

The API includes optional re-ranking using `BAAI/bge-reranker-v2-m3` to improve result quality.

**How it works:**
1. Retrieve top 25 passages from FAISS (fast, approximate)
2. Re-rank using cross-encoder model (slower, more accurate)
3. Return top k results

**Usage:**
```bash
# With re-ranking (default)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 10, "rerank": true}'

# Without re-ranking
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 10, "rerank": false}'
```

Response includes both `score` (FAISS similarity) and `rerank_score` (cross-encoder score).

[benchmark link](https://claude.ai/public/artifacts/44a9a06f-3860-48cc-8e91-bf4b20c91ce5)

## Models and Data

This system builds on third-party models, each under its own license. Verify these
before use — particularly for commercial deployments:

- **[facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco)** —
  passage embeddings. The upstream
  [facebookresearch/contriever](https://github.com/facebookresearch/contriever)
  repository is released under **CC BY-NC 4.0 (non-commercial)**.
- **[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)** —
  cross-encoder re-ranking.
- **[FAISS](https://github.com/facebookresearch/faiss)** — MIT licensed.

The indexed corpus is not distributed with this repository.

## Citation

This work builds on the retrieval setup described in:

> Xinxi Lyu, Michael Duan, Rulin Shao, Pang Wei Koh, and Sewon Min.
> *Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks.*
> arXiv:2507.01297, 2025.

```bibtex
@article{lyu2025frustratingly,
  title   = {Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks},
  author  = {Lyu, Xinxi and Duan, Michael and Shao, Rulin and Koh, Pang Wei and Min, Sewon},
  journal = {arXiv preprint arXiv:2507.01297},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.01297}
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the
full text.

Note that the Contriever model weights this system depends on carry a
non-commercial license of their own — see [Models and Data](#models-and-data).
