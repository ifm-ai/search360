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
