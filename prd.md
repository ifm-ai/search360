Question: Suppose you have:
Original Document JSONL Files:
{”id”: 0, “text”: “full text about Paris is the capital of France”}
{"id": 1, "text": "Full text about the London is the capital of England"}
JSONL Files: passages/wiki.jsonl: {"id": 0, "text": "Paris is the capital of France"} {"id": 1, "text": "London is the capital of England"}
passages/books.jsonl: {"id": 0, "text": "The quick brown fox jumps"} {"id": 1, "text": "Machine learning is a subset of AI"}
After Indexing, the arrays would be:
passage_filenames: maps shard_id -> filename
passage_filenames = np.array(["wiki.jsonl", "books.jsonl"])
index_id_to_file_id: maps each passage's index_id -> shard_id
index_id_to_file_id = np.array([ 0, # index_id 0 -> shard 0 (wiki.jsonl) 0, # index_id 1 -> shard 0 (wiki.jsonl) 1, # index_id 2 -> shard 1 (books.jsonl) 1 # index_id 3 -> shard 1 (books.jsonl) ])
psg_pos_id_array: byte position where each passage starts
psg_pos_id_array = np.array([ 0, # index_id 0 starts at byte 0 in wiki.jsonl 47, # index_id 1 starts at byte 47 in wiki.jsonl 0, # index_id 2 starts at byte 0 in books.jsonl 41 # index_id 3 starts at byte 41 in books.jsonl ])
Step-by-Step Example
Call: getpassage(index_id=3)
def getpassage(self, index_id): # index_id = 3

  # Step 1: Find which file contains this passage
  file_shard_id = self.index_id_to_file_id[index_id]
  # = self.index_id_to_file_id[3] = 1
  filename = self.passage_filenames[file_shard_id]
  # = self.passage_filenames[1] = "books.jsonl"
  # Step 2: Get exact byte position in that file
  position = self.psg_pos_id_array[index_id]
  # = self.psg_pos_id_array[3] = 41
  # Step 3: Open file and seek directly to position
  with open("passages/books.jsonl", 'r') as file:
      file.seek(41)  # Jump to byte 41
      # File cursor is now at: ...jumps"}\\n|{"id": 1, "text": "Machine learning...
      #                                    ^ cursor here (byte 41)
      line = file.readline()  # Read until newline
      # line = '{"id": 1, "text": "Machine learning is a subset of AI"}\\n'
  return json.loads(line)
  # Returns: {"id": 1, "text": "Machine learning is a subset of AI"}

Imagine now I want to add the lookup of this original document given the passage as well. What mappings will I need?

Answer: Great question! To map from a passage back to its original document, you'll need additional mapping arrays. Here's what you should add:
New Mappings Needed
python# Map each passage's index_id -> original document ID
passage_to_doc_id = np.array([
    0,  # index_id 0 (Paris passage) -> doc_id 0
    1,  # index_id 1 (London passage) -> doc_id 1
    0,  # index_id 2 (fox passage) -> doc_id 0 (if from same doc)
    1   # index_id 3 (ML passage) -> doc_id 1 (if from same doc)
])

# Map each document ID -> its filename
doc_filenames = np.array(["wiki_docs.jsonl", "books_docs.jsonl"])

# Map each document ID -> which shard it belongs to
doc_id_to_file_id = np.array([
    0,  # doc_id 0 -> shard 0 (wiki_docs.jsonl)
    1   # doc_id 1 -> shard 1 (books_docs.jsonl)
])

# Map each document ID -> byte position in its file
doc_pos_id_array = np.array([
    0,   # doc_id 0 starts at byte 0
    59   # doc_id 1 starts at byte 59
])
Updated get_passage Function
pythondef get_passage_with_document(self, index_id):
    # Get the passage (existing logic)
    file_shard_id = self.index_id_to_file_id[index_id]
    filename = self.passage_filenames[file_shard_id]
    position = self.psg_pos_id_array[index_id]
    
    with open(f"passages/{filename}", 'r') as file:
        file.seek(position)
        passage = json.loads(file.readline())
    
    # NEW: Get the original document
    doc_id = self.passage_to_doc_id[index_id]
    doc_shard_id = self.doc_id_to_file_id[doc_id]
    doc_filename = self.doc_filenames[doc_shard_id]
    doc_position = self.doc_pos_id_array[doc_id]
    
    with open(f"documents/{doc_filename}", 'r') as file:
        file.seek(doc_position)
        document = json.loads(file.readline())
    
    return {
        'passage': passage,
        'document': document
    }
Key Design Decision
The most important new mapping is passage_to_doc_id - this creates the link between passages and their source documents. The other arrays (doc_filenames, doc_id_to_file_id, doc_pos_id_array) simply mirror your existing passage lookup structure but for documents.
This design assumes each passage comes from exactly one document, which is typical for retrieval systems.

User: given the Original Document JSONL Files can you help me create the doc_id_to_file_id and doc_pos_id_array numpy files?

Absolutely! Here's a script to create those numpy arrays from your original document JSONL files:
@01_map_documents.py


---

## FAISS Integration: Critical Ordering Requirements

### The Flow
```
FAISS Index → passage_id → Passage Lookup → doc_id → Document Lookup
```

### Critical Invariant
**passage_id == array_index == faiss_id**

When FAISS returns an integer `faiss_id` from search, this number MUST be the exact index into your numpy mapping arrays.

### Key Requirements

1. **FAISS IDs = Array Indices**
   - FAISS returns `faiss_id=42` → this is `passage_id=42` in all arrays
   - `passage_id_to_file_id[42]`, `passage_pos_id_array[42]`, `passage_to_doc_id[42]` all correspond to the same passage

2. **Embedding Order = Passage Creation Order**
   - Embeddings MUST be added to FAISS in the EXACT same order as passages were written
   - The `create_passages.py` script processes files in **sorted order**: `sorted(Path(document_dir).glob("*.jsonl"))`
   - When creating embeddings, iterate through passages sequentially from passage_id=0 to passage_id=N-1

3. **Sequential Passage IDs**
   - passage_id starts at 0
   - Increments by 1 for each passage across ALL files
   - Total passages = len(passage_id_to_file_id)

4. **Array Lengths Must Match**
   ```
   len(passage_id_to_file_id) == len(passage_pos_id_array) == len(passage_to_doc_id) == faiss_index.ntotal
   ```

### What Can Go Wrong

❌ **Processing files in different order when creating embeddings** → FAISS IDs won't match passage IDs
❌ **Skipping passages when embedding** → Offset in all subsequent IDs
❌ **Reordering embeddings before adding to FAISS** → FAISS IDs point to wrong passages
❌ **Parallel embedding without maintaining order** → Random passage order

### Retrieval Pipeline

```
1. Query FAISS → Get faiss_id (e.g., 42)
2. Use faiss_id as passage_id to lookup passage:
   - file_id = passage_id_to_file_id[42]
   - filename = passage_filenames[file_id]
   - position = passage_pos_id_array[42]
   - Read passage from file at position
3. Get source document:
   - doc_id = passage_to_doc_id[42]
   - Use doc_id to lookup in doc_id_to_file_id, doc_pos_id_array, doc_filenames
   - Read document from file at position
```

### Bottom Line

**When creating embeddings for FAISS, iterate through passages in sequential order (passage_id 0, 1, 2, ..., N-1) and add embeddings to FAISS in that exact same order. Never shuffle, skip, or reorder.**

