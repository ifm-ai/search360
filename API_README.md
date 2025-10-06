# FAISS Passage Retrieval API

FastAPI service for searching passages and retrieving source documents.

## Features

- ✅ **Fast search** with FAISS index (GPU/CPU support)
- ✅ **Passage retrieval** with full metadata
- ✅ **Document ranking** by occurrence count
- ✅ **Flexible responses** (passages only or passages + documents)
- ✅ **Health checks** and statistics endpoints

## Quick Start

### 1. Start the server (CPU):
```bash
python api.py
```

### 2. Start the server (GPU):
```bash
python api.py --use_gpu
python -m api.server --use_gpu
```

### 3. Custom configuration:
```bash
python api.py \
    --use_gpu \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
```

## API Endpoints

### POST `/search`

Search for passages and optionally return ranked source documents.

**Request:**
```bash
curl -X POST http://0.0.0.0:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 10,
    "return_fulltext": false
  }'
```

**Response (return_fulltext=false):**
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "rank": 1,
      "score": 0.8945,
      "passage_id": 12345,
      "passage_text": "Machine learning is a subset of AI...",
      "passage_file": "arxiv_part_0001_passages.jsonl",
      "passage_position": 123456,
      "doc_id": 456
    }
  ],
  "documents": null
}
```

**Response (return_fulltext=true):**
```json
{
  "query": "What is machine learning?",
  "results": [...],
  "documents": [
    {
      "rank": 1,
      "doc_id": 456,
      "occurrences": 3,
      "doc_text": "Full document text...",
      "doc_file": "arxiv_part_0001.jsonl",
      "doc_position": 789
    }
  ]
}
```

### POST `/get_document`

Retrieve a document by document ID.

**Request:**
```bash
curl -X POST http://0.0.0.0:8000/get_document \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": 33015280
  }' \
  | jq .
```

**Response:**
```json
{
  "doc_id": 456,
  "doc_text": "Full document text...",
  "doc_file": "arxiv_part_0001.jsonl",
  "doc_position": 789
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "index_loaded": true
}
```

### GET `/stats`

Get index statistics.

**Response:**
```json
{
  "total_passages": 1600000000,
  "total_documents": 145000000,
  "num_passage_files": 1401,
  "num_document_files": 1401
}
```

## Usage Examples

### cURL

**Basic search:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 5}'
```

**Search with full documents:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum physics", "k": 10, "return_fulltext": true}'
```

**Get document:**
```bash
curl -X POST "http://localhost:8000/get_document" \
  -H "Content-Type: application/json" \
  -d '{"doc_id": 456}'
```

**Health check:**
```bash
curl "http://localhost:8000/health"
```

### Python

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "What is machine learning?",
        "k": 5,
        "return_fulltext": False
    }
)
results = response.json()

# Get document
doc_response = requests.post(
    "http://localhost:8000/get_document",
    json={"doc_id": 456}
)
document = doc_response.json()
```

### JavaScript

```javascript
// Search
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'What is machine learning?',
    k: 5,
    return_fulltext: false
  })
});
const results = await response.json();
```

## Configuration

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--index_path` | Path to index | FAISS index file |
| `--output_dir` | Path to outputs | Mappings directory |
| `--passages_dir` | Path to passages | Passage JSONL files |
| `--documents_dir` | Path to documents | Document JSONL files |
| `--model_name` | `facebook/contriever` | HuggingFace model |
| `--nprobe` | `2048` | FAISS nprobe value |
| `--use_gpu` | `False` | Enable GPU acceleration |
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8000` | Server port |
| `--workers` | `1` | Number of workers |

### Limits

- **Max k**: 100 (configurable in `api.py`)
- **Empty queries**: Returns empty results list

## Interactive API Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Document Ranking

When `return_fulltext=true`, documents are ranked by **occurrence count** (how many passages came from that document).

### Current Algorithm (in `api_utils.py`)

```python
rank_documents_by_occurrence(passage_results)
```

Pure count-based: More passages from a document = higher rank.

### Alternative Algorithms (available but not used)

1. **Score-based**: `rank_documents_by_score_sum()` - Sum of passage scores
2. **Best passage**: `rank_documents_by_best_passage()` - Highest-ranked passage

To change: Edit `api.py` line with `rank_documents_by_occurrence()`.

## Performance

- **GPU (8x GPUs)**: ~10-50ms per query
- **CPU**: ~100-500ms per query
- **Startup time**: ~30-60 seconds (loading index + model)

## Troubleshooting

**Server won't start:**
- Check if port 8000 is available: `lsof -i :8000`
- Try different port: `python api.py --port 8001`

**GPU not detected:**
- Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check FAISS GPU support: `python -c "import faiss; print(faiss.get_num_gpus())"`

**Out of memory:**
- Reduce workers: `--workers 1`
- Use CPU: Remove `--use_gpu` flag

## Development

Install dependencies:
```bash
pip install fastapi uvicorn torch transformers faiss-gpu numpy
```

Run in development mode:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```


## Some stats

For 8 workers, the service is using 303GB of memory and 54GB VRAM per GPU.
For 16 workers, the service will use 600GB of memory and 108GB VRAM per GPU.

This is doable. Lets test this.

## Running with SLURM

### Starting the service

Submit the service to SLURM:
```bash
sbatch slurm_start_service.sh
```

The script will:
1. Request a full node (8 GPUs, 1024GB RAM, 128 CPUs)
2. Start the FastAPI service with 16 workers
3. Create a server info file at `runtime/server_info_<job_id>.json`
4. Poll the health endpoint until ready
5. Run until manually stopped or time limit reached

### Server Info File

The SLURM script creates a JSON file to track server status:

**Location:** `runtime/server_info_<job_id>.json`

**Format:**
```json
{
  "host": "fs-mbz-gpu-757",
  "port": 8000,
  "url": "http://fs-mbz-gpu-757:8000",
  "job_id": "12345",
  "status": "ready",
  "started_at": "2025-10-06T10:30:00+00:00",
  "ready_at": "2025-10-06T10:31:45+00:00"
}
```

**Status values:**
- `loading`: Server is starting up, index being loaded
- `ready`: Server is ready to accept requests

The file is automatically deleted when the job ends.

### Using the server info from another script

**Python example:**
```python
import json
import time
from pathlib import Path

def wait_for_server(job_id, timeout=300):
    """Wait for server to be ready and return connection info."""
    server_file = Path(f"runtime/server_info_{job_id}.json")

    start = time.time()
    while time.time() - start < timeout:
        if server_file.exists():
            with open(server_file) as f:
                info = json.load(f)
                if info["status"] == "ready":
                    print(f"Server ready at: {info['url']}")
                    return info
                else:
                    print(f"Server status: {info['status']}, waiting...")
        time.sleep(2)

    raise TimeoutError(f"Server not ready after {timeout}s")

# Usage
job_id = "12345"  # From sbatch output
server_info = wait_for_server(job_id)
url = server_info["url"]

# Now you can make requests
import requests
response = requests.post(
    f"{url}/search",
    json={"query": "What is machine learning?", "k": 10}
)
```

**Bash example:**
```bash
#!/bin/bash
JOB_ID=$1
SERVER_FILE="runtime/server_info_${JOB_ID}.json"

# Wait for server to be ready
echo "Waiting for server..."
while [ ! -f "$SERVER_FILE" ] || [ "$(jq -r '.status' $SERVER_FILE)" != "ready" ]; do
    sleep 2
done

# Extract URL
URL=$(jq -r '.url' $SERVER_FILE)
echo "Server ready at: $URL"

# Make request
curl -X POST "$URL/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 10}'
```

### Finding running servers

List all active server info files:
```bash
ls -la runtime/server_info_*.json
```

Check status of a specific job:
```bash
cat runtime/server_info_12345.json | jq .
```

### Stopping the service

```bash
scancel <job_id>
```

The cleanup trap will automatically remove the server info file. 