# FastGraphRAG Hyperparameters

**Document:** Comprehensive reference for all FastGraphRAG configuration parameters
**Target:** Ollama local models

**Note:** `run_fast-graphrag.py` uses library defaults for all parameters except where noted.

---

## Current Configuration Summary

| Hyperparameter | Value in `run_fast-graphrag.py` |
|----------------|----------------------------------|
| `max_gleaning_steps` | **1** ✅ (explicitly configured) |
| `chunk_token_size` | 600 (updated from default 800) |
| `chunk_token_overlap` | 100 (default) |
| `separators` | Default list (default) |
| `entities_max_tokens` | 4000 (default) |
| `relations_max_tokens` | 3000 (default) |
| `chunks_max_tokens` | 9000 (default) |
| `entity_ranking_policy.threshold` | 0.005 (default) |
| `relation_ranking_policy.top_k` | 64 (default) |
| `chunk_ranking_policy.top_k` | 8 (default) |
| `insert_similarity_score_threshold` | 0.9 (default) |
| `node_summarization_ratio` | 0.5 (default) |
| `n_checkpoints` | 0 (default) |
| `max_requests_concurrent` | 1024 (default from env) |
| `llm_retry_attempts` | 5 (updated from default 3) |

**Total context budget (query):** 16,000 tokens (4,000 entities + 3,000 relations + 9,000 chunks)

---

## Chunking Hyperparameters

### `chunk_token_size`
**Location:** `fast_graphrag/_services/_chunk_extraction.py:32`
**Type:** `int`
**Default:** `600` (updated from 800)
**Current in `run_fast-graphrag.py`:** `600` (uses default)
**Units:** tokens

**Description:** Maximum size of each text chunk before splitting.

**Impact:**
- Larger values → Fewer chunks, more context per extraction, longer processing per chunk
- Smaller values → More chunks, less context, faster per chunk but more total overhead

**Recent Change:**
- Reduced from 800 to 600 tokens to improve small model (gemma3:12b) reliability
- Smaller chunks reduce cognitive load and improve structured output adherence

**Recommendations by Model Size:**
| Model Size | Recommended Value | Rationale |
|------------|-------------------|-----------|
| 0.5B - 1B | 400-600 | Reduce cognitive load on small models |
| 3B - 7B | 600-800 | Balanced context and speed |
| 7B - 12B | 600 (current) | Optimal for gemma3:12b reliability |
| 14B+ | 800-1200 | Maximize context, reduce chunk count |

**Example:**
```python
from fast_graphrag._services import DefaultChunkingService, DefaultChunkingServiceConfig

config = DefaultChunkingServiceConfig(
    chunk_token_size=800  # Default
)
```

---

### `chunk_token_overlap`
**Location:** `fast_graphrag/_services/_chunk_extraction.py:33`
**Type:** `int`
**Default:** `100`
**Current in `run_fast-graphrag.py`:** `100` (uses default)
**Units:** tokens

**Description:** Number of tokens to overlap between consecutive chunks for context continuity.

**Impact:**
- Larger overlap → Better boundary entity/relationship capture, more redundancy
- Smaller overlap → Less redundancy, faster, may miss boundary relationships

**Recommendations:**
- **Keep default (100 tokens)** for most use cases
- Medical/technical domains: 100-150 tokens (entities can be long)
- Simple text: 50-100 tokens

**Trade-off Analysis:**
| Overlap | Context Preservation | Token Overhead | Use Case |
|---------|---------------------|----------------|----------|
| 0 | Poor | 0% | Not recommended |
| 50 | Fair | 6% | Simple documents only |
| 100 (default) | Good | 12.5% | General purpose ✅ |
| 150 | Excellent | 18.75% | Complex technical text |
| 200 | Maximum | 25% | Only if critical relationships missed |

---

### `separators`
**Location:** `fast_graphrag/_services/_chunk_extraction.py:13-26`
**Type:** `List[str]`
**Default:** `["\n\n\n", "\n\n", "\r\n\r\n", "。", "．", ".", "！", "!", "？", "?"]`
**Current in `run_fast-graphrag.py`:** Uses default

**Description:** Ordered list of text separators used for chunk splitting (priority order).

**Impact:** Determines natural breaking points (paragraphs > sentences > punctuation)

**Customization:** Add domain-specific separators
```python
MEDICAL_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\nDiagnosis:",
    "\nTreatment:",
    "\nHistory:",
    ".",
    "!"
]
```

---

## Information Extraction Hyperparameters

### `max_gleaning_steps`
**Location:** `fast_graphrag/_services/_base.py:46`
**Type:** `int`
**Default:** `0` (disabled by default)
**Current in `run_fast-graphrag.py`:** `1` (enabled)
**Range:** 0-3

**Description:** Number of additional extraction passes to capture missed entities/relationships.

**How it works:**
1. Initial extraction completes
2. LLM is asked: "MANY entities were missed, add them below"
3. Additional entities/relationships added to graph
4. Process repeats up to `max_gleaning_steps` times

**Impact per iteration:**
- Adds ~2x tokens per gleaning iteration
- Increases processing time proportionally
- Improves completeness by ~5-10% per iteration

**Token Impact:**
| Gleaning Steps | Tokens/Chunk | Time (3B model) | Completeness |
|----------------|--------------|-----------------|--------------|
| 0 | 3,510 | 47 sec | Baseline |
| 1 | 7,640 | 1.7 min | +5-10% |
| 2 | 12,200 | 2.7 min | +10-15% |
| 3 | 16,700 | 3.7 min | +15-20% |

**Recommendations:**
- **0 (library default):** Fast development/testing, simple documents
- **1 (production default in `run_fast-graphrag.py`):** Production balance (speed + quality) ✅ **CURRENT**
- **2:** High-quality extraction, complex documents
- **3:** Maximum completeness, research applications

**Example:**
```python
from fast_graphrag._services import DefaultInformationExtractionService

# In run_fast-graphrag.py (current implementation)
config=GraphRAG.Config(
    llm_service=llm_service,
    embedding_service=embedding_service,
    information_extraction_service_cls=lambda: DefaultInformationExtractionService(
        graph_upsert=GraphRAG.Config().information_extraction_upsert_policy,
        max_gleaning_steps=1  # CURRENT: Enabled with 1 iteration
    ),
)
```

---

## Query Hyperparameters

### `entities_max_tokens`
**Location:** `fast_graphrag/_graphrag.py:29`
**Type:** `int`
**Default:** `4000`
**Current in `run_fast-graphrag.py`:** `4000` (uses default)
**Units:** tokens

**Description:** Maximum tokens for entity context in query response.

**Impact:** Controls how many entities are included in the context window for answer generation.

---

### `relations_max_tokens`
**Location:** `fast_graphrag/_graphrag.py:30`
**Type:** `int`
**Default:** `3000`
**Current in `run_fast-graphrag.py`:** `3000` (uses default)
**Units:** tokens

**Description:** Maximum tokens for relationship context in query response.

---

### `chunks_max_tokens`
**Location:** `fast_graphrag/_graphrag.py:31`
**Type:** `int`
**Default:** `9000`
**Current in `run_fast-graphrag.py`:** `9000` (uses default)
**Units:** tokens

**Description:** Maximum tokens for chunk (raw text) context in query response.

**Total context budget:** 4000 + 3000 + 9000 = **16,000 tokens**

---

## Ranking Hyperparameters

### `entity_ranking_policy.threshold`
**Location:** `fast_graphrag/__init__.py:80`
**Type:** `float`
**Default:** `0.005`
**Current in `run_fast-graphrag.py`:** `0.005` (uses default)
**Range:** 0.0 - 1.0

**Description:** Minimum PageRank score threshold for entities to be included in query results.

**Impact:**
- Higher values → Fewer entities, more focused results
- Lower values → More entities, broader context

---

### `relation_ranking_policy.top_k`
**Location:** `fast_graphrag/__init__.py:83`
**Type:** `int`
**Default:** `64`
**Current in `run_fast-graphrag.py`:** `64` (uses default)
**Range:** 1-∞

**Description:** Maximum number of top-ranked relationships to include in query results.

---

### `chunk_ranking_policy.top_k`
**Location:** `fast_graphrag/__init__.py:86`
**Type:** `int`
**Default:** `8`
**Current in `run_fast-graphrag.py`:** `8` (uses default)
**Range:** 1-∞

**Description:** Maximum number of top-ranked chunks to include in query results.

---

## Storage & Deduplication Hyperparameters

### `insert_similarity_score_threshold`
**Location:** `fast_graphrag/_services/_state_manager.py` (DefaultStateManagerService)
**Type:** `float`
**Default:** `0.9`
**Current in `run_fast-graphrag.py`:** `0.9` (uses default)
**Range:** 0.0 - 1.0

**Description:** Cosine similarity threshold for entity deduplication during insertion.

**How it works:**
- After inserting entities, system checks for similar entities using vector embeddings
- If similarity > threshold, entities are linked with "is" relationship
- Prevents duplicate entities with slightly different names

**Impact:**
- Higher values (0.95+) → Stricter matching, fewer merges
- Lower values (0.85-) → More aggressive merging, may incorrectly link different entities

**Recommendations:**
- **0.9 (default):** Good balance ✅
- **0.95:** Very strict (different spellings treated as separate)
- **0.85:** Aggressive (useful for noisy text with typos)

---

### `node_summarization_ratio`
**Location:** `fast_graphrag/_policies/_graph_upsert.py` (NodeUpsertPolicy_SummarizeDescription.Config)
**Type:** `float`
**Default:** `0.5`
**Range:** 0.0 - 1.0

**Description:** Ratio to determine when to summarize node descriptions (if they grow too long).

**Impact:** Controls when LLM is called to summarize accumulated entity descriptions.

---

## Checkpoint Hyperparameters

### `n_checkpoints`
**Location:** `fast_graphrag/_graphrag.py:42`
**Type:** `int`
**Default:** `0` (disabled)
**Current in `run_fast-graphrag.py`:** `0` (uses default)
**Range:** 0-∞

**Description:** Number of historical checkpoints to keep for rollback capability.

**Impact:**
- 0: No checkpoints (default)
- N > 0: Keep last N versions of graph state

**Use case:** Allows reverting to previous state if data corruption occurs.

---

## LLM Service Hyperparameters (Ollama-specific)

### `llm_retry_attempts`
**Location:** `fast_graphrag/_llm/_ollama.py:201`
**Type:** `int`
**Default:** `5` (updated from 3)
**Range:** 1-10

**Description:** Maximum number of retry attempts for LLM requests when validation or network errors occur.

**Impact:**
- Higher values → Better recovery from transient errors, longer wait times on persistent failures
- Lower values → Faster failure detection, less tolerance for intermittent issues

**Recent Change:**
- Increased from 3 to 5 attempts to handle small model (gemma3:12b) inconsistency
- Combined with auto-fix mechanism for Graph JSON validation errors

**Retry Strategy:**
- Exponential backoff: 4s → 8s → 16s → 30s (max)
- Retries on: network errors, timeouts, JSON validation failures
- Auto-fix applied after validation errors before retrying

---

### `max_requests_concurrent`
**Location:** `fast_graphrag/_llm/_base.py:73`
**Type:** `int`
**Default:** From `CONCURRENT_TASK_LIMIT` env var (default: 1024)
**Range:** 1-∞

**Description:** Maximum number of concurrent LLM requests.

**Recommendations for Ollama:**
- **1-4:** CPU-based inference (prevents overwhelming)
- **8-16:** GPU-based inference (single GPU)
- **32+:** Multi-GPU setups

**Example:**
```bash
export CONCURRENT_TASK_LIMIT=4
```

---

## Performance Tuning Guide

### Quick Reference Table

| Use Case | chunk_token_size | chunk_token_overlap | max_gleaning_steps | Model |
|----------|------------------|---------------------|-------------------|-------|
| **Dev/Testing** | 400-600 | 50 | 0 | 3B |
| **Production (Speed)** | 800-1000 | 150 | 1 | 14B+ |
| **Production (Quality)** | 600-800 | 100-150 | 2 | 14B+ |
| **Production (Small Model)** | 600 (current) | 100 | 1 | 7B-12B |
| **Research (Max Quality)** | 800 | 150 | 3 | 14B+ |

---

### Expected Processing Times (50,000 char document)

| Configuration | Tokens/Doc | Time (3B) | Time (7B) |
|---------------|-----------|-----------|-----------|
| chunk=400, gleaning=0 | ~60,000 | 13 min | 7 min |
| chunk=600, gleaning=0 | ~62,000 | 14 min | 7 min |
| chunk=600, gleaning=1 | **~135,000** | **30 min** | **15 min** |
| chunk=800, gleaning=0 | ~63,000 | 14 min | 7 min |
| chunk=800, gleaning=1 | ~137,000 | 31 min | 15 min |
| chunk=800, gleaning=2 | ~220,000 | 49 min | 24 min |
| chunk=1200, gleaning=1 | ~91,000 | 20 min | 10 min |

---

## Configuration Best Practices

### 1. Start with Defaults
```python
# Use defaults for initial testing
grag = GraphRAG(
    working_dir="./output",
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
)
```

### 2. Enable Gleaning for Production
```python
# Add 1 gleaning iteration for better quality
config = GraphRAG.Config(
    information_extraction_service_cls=lambda: DefaultInformationExtractionService(
        max_gleaning_steps=1
    )
)
```

### 3. Optimize Chunking for Larger Models
```python
# For 7B+ models, increase chunk size
config.chunking_service_cls = lambda: DefaultChunkingService(
    config=DefaultChunkingServiceConfig(
        chunk_token_size=1200,
        chunk_token_overlap=150
    )
)
```

---

## Monitoring & Debugging

### Key Metrics to Track

1. **Processing time per document**
2. **Average tokens per chunk**
3. **Entities extracted per document**
4. **Relationships extracted per document**
5. **Pydantic validation failure rate**

### Debug Flags

**Enable verbose logging:**
```python
import logging
logging.getLogger("graphrag").setLevel(logging.DEBUG)
```

---

## Summary: Critical Hyperparameters

| Parameter | Default | Current (`run_fast-graphrag.py`) | Impact Level | Tune For |
|-----------|---------|----------------------------------|--------------|----------|
| `chunk_token_size` | 600 | 600 | ⭐⭐⭐ High | Model size, reliability |
| `chunk_token_overlap` | 100 | 100 | ⭐⭐⭐ High | Accuracy |
| `max_gleaning_steps` | 0 | **1** ✅ | ⭐⭐⭐ High | Quality vs Speed |
| `llm_retry_attempts` | 5 | 5 | ⭐⭐ Medium | Reliability |
| `entities_max_tokens` | 4000 | 4000 | ⭐ Low | Query context |
| `entity_ranking_policy.threshold` | 0.005 | 0.005 | ⭐⭐ Medium | Precision/Recall |
| `insert_similarity_score_threshold` | 0.9 | 0.9 | ⭐⭐ Medium | Deduplication |

**Most important:** `max_gleaning_steps` (currently enabled in production script)

**Current production configuration:**
- Gleaning enabled with 1 iteration for improved quality
- Chunk size reduced to 600 tokens for small model reliability
- LLM retries increased to 5 with auto-fix for validation errors
