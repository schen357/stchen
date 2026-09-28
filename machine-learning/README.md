# Machine Learning: Flowchart → RAG Assistant

The goal: **chat with an LLM that uses retrieval-augmented generation (RAG) to answer questions about specialized processes, where those processes start out as flowchart images.**

The work is split into small, independent stages. Each one is its own `uv` project with its own dependencies. They pass data to each other through plain files (pickle and parquet), so you can run, test, and change each stage without touching the others.

```
 ┌──────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
 │  Flowchart       │     │  image_to_text       │     │  api_generation      │     │  rag_sources     │
 │  images (.jpg)   │ ──▶ │  Qwen2.5-VL (local)  │ ──▶ │  Claude API          │ ──▶ │  ChromaDB +      │
 │                  │     │  image → steps       │     │  process → Q set     │     │  cross-encoder   │
 └──────────────────┘     └──────────────────────┘     └──────────────────────┘     └────────┬─────────┘
                           flowchart_instructions.pkl    question_set.pq                     │
                                                                                             ▼
                                                          ┌──────────────────────────────────────────┐
                                                          │  api_generation/chatbot.py (planned)     │
                                                          │  Claude + retrieved context → answers    │
                                                          └──────────────────────────────────────────┘
```

---

## Contents

| Directory | Purpose | Key tech | Status |
|---|---|---|---|
| [`image_to_text/`](image_to_text/) | Turns flowchart images into written, step-by-step instructions using an open-source vision-language model (VLM) | Hugging Face `transformers`, Qwen2.5-VL-3B-Instruct, PyTorch (MPS) | Working, with unit tests |
| [`api_generation/`](api_generation/) | Calls the Anthropic API to create a set of questions for each process, then (planned) answers questions with RAG context | `anthropic`, `pandas`, parquet | Question generation working, with unit tests; chatbot is a stub |
| [`rag_sources/`](rag_sources/) | Builds a local vector store of processes and runs a two-stage retrieval: broad semantic search, then reranking | `chromadb`, `sentence-transformers` cross-encoder | Prototype |
| [`mcp/`](mcp/) | Reserved for a future Model Context Protocol (MCP) server that exposes the pipeline as tools | — | Placeholder |

---

## Stage 1: `image_to_text`: flowcharts → instructions

**Entry point:** `image_to_text/image_to_text/flowchart_to_instructions.py`

Runs the open-source VLM **[Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)** on your machine and prompts it to rewrite each flowchart as clear, specific instructions, with every decision branch and its criteria spelled out.

- **Dataset:** the Roboflow [FlowChart v3](https://universe.roboflow.com/flowchart-zdtle/flowchart-hdqrb) dataset (CC BY 4.0, 640×640 images). A sample of the `train/` split is in `image_to_text/flowcharts/FlowChart.v3i.multiclass/`.
- **How it works:**
  1. `image_paths()` collects `.jpg`/`.jpeg`/`.png` files from a dataset split. Paths are resolved relative to the package, so the script works from any directory.
  2. `image_to_instructions()` builds one Qwen chat prompt per image, runs a batched processor call, and generates deterministically (`do_sample=False`, `max_new_tokens=512`) under `torch.inference_mode()`.
  3. The prompt tokens are trimmed off the output, and only the newly generated text is decoded.
- **Hardware:** set up for Apple Silicon (`DEVICE = "mps"`, `float16`). Images are downscaled with `max_pixels=256*256` to keep memory use down.
- **Output:** `flowchart_instructions.pkl`, a `list[str]` with one set of instructions per image.
- **Container:** includes a `Dockerfile` (Python 3.11-slim + `uv`) for running outside macOS. Note that `mps` isn't available in Docker, so change `DEVICE` to `cpu`/`cuda` there.

## Stage 2: `api_generation`: synthetic question sets with Claude

**Entry point:** `api_generation/api_generation/generate_question_set.py`

For each process from Stage 1, Claude (`claude-sonnet-5`, set in `utils.py`) is asked for:
- a short, unique **process title** (e.g. *"Order Quality Assurance"*), and
- as many **questions as possible that the process can answer**.

The answer is requested as JSON, parsed by `parse_questions()`, and collected into a DataFrame:

| column | type | description |
|---|---|---|
| `process_title` | `str` | Short title created by Claude, also used as the document ID in ChromaDB |
| `process` | `str` | The full written instructions from Stage 1 |
| `question_set` | `list[str]` | Synthetic questions this process can answer |

**Output:** `question_set.pq` (parquet).

These synthetic questions have two uses: (1) they're an evaluation set for retrieval ("does question *q* retrieve the process it came from?"), and (2) they can be added to documents to improve semantic matching.

**Planned:** `chatbot.py` will run the same questions through Claude **with** and **without** retrieved context, to measure how much RAG actually helps.

## Stage 3: `rag_sources`: retrieval

**Entry point:** `rag_sources/rag_sources/chromadb_rerank.py`

Two-stage retrieval over the processes:

1. **Broad recall.** Processes are upserted into a persistent ChromaDB collection (`processes`, stored in `rag_sources/my_chroma_db/`), with `process_title` as the ID. ChromaDB embeds the documents automatically. A query returns the top-*k* nearest neighbors.
2. **Precise reranking.** Each `(query, document)` pair is scored by the cross-encoder [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base), and results are re-sorted by relevance.

Bi-encoder search is fast but coarse. A cross-encoder is slower but reads the query and the document together, so using it only on the shortlist gets you most of the accuracy for little extra cost.

`build_index.py` is a small FAISS `IndexFlatL2` example on random vectors, kept as a reference for comparing against ChromaDB.

## `mcp`

Placeholder for exposing the retriever and assistant as MCP tools, so any MCP-compatible client can query the processes.

---

## Getting started

### Prerequisites

- Python **3.11–3.13**
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- An Anthropic API key for Stage 2
- Apple Silicon Mac for Stage 1 as configured (or edit `DEVICE`)

Each subproject has its own virtual environment, so run commands from inside that subproject's directory.

### 1. Flowcharts → instructions

```bash
cd image_to_text
uv sync
uv run python image_to_text/flowchart_to_instructions.py
# → writes flowchart_instructions.pkl to the current directory
```

The first run downloads the Qwen2.5-VL weights (several GB) from Hugging Face.

### 2. Instructions → question sets

```bash
cp image_to_text/flowchart_instructions.pkl api_generation/api_generation/

cd api_generation
export ANTHROPIC_API_KEY=<your key>
uv sync
uv run python api_generation/generate_question_set.py
# → writes api_generation/question_set.pq (next to the module)
```

### 3. Index and retrieve

```bash
cp api_generation/api_generation/question_set.pq rag_sources/rag_sources/

cd rag_sources
uv sync
uv run python rag_sources/chromadb_rerank.py
# → builds/updates ./my_chroma_db, runs a sample query, prints raw and reranked results
```

### Running tests

```bash
cd image_to_text && uv run pytest     # model and processor are mocked, no weights needed
cd api_generation && uv run pytest    # Anthropic client is faked, no API key or network needed
```

---

## Data flow at a glance

| Artifact | Produced by | Consumed by | Format |
|---|---|---|---|
| `FlowChart.v3i.multiclass/train/*.jpg` | Roboflow export | `image_to_text` | JPEG |
| `flowchart_instructions.pkl` | `image_to_text` | `api_generation` | pickled `list[str]` |
| `question_set.pq` | `api_generation` | `rag_sources`, (planned) chatbot | parquet |
| `my_chroma_db/` | `rag_sources` | retrieval / chatbot | ChromaDB (SQLite + HNSW) |

Stages hand off artifacts by copying files by hand (see *Getting started*). This is deliberate for now: you can inspect and re-run any stage on its own without re-running expensive steps before it, such as VLM inference.

---

## Known gaps and next steps

- [ ] **`build_index.py`**: imports `faiss`, which isn't listed in `rag_sources/pyproject.toml`.
- [ ] **Retrieval with questions**: add each process's `question_set` to its document (or index questions as separate entries that point back to the process), then measure the recall gain.
- [ ] **Retrieval evaluation**: use the synthetic questions to compute hit@k / MRR, with and without reranking.
- [ ] **Chatbot**: implement `chatbot.py` to compare Claude answers with RAG vs. without.
- [ ] **MCP server**: expose `search_processes` / `ask_about_process` tools.
- [ ] **Artifact paths**: replace manual copies with a shared `data/` directory or CLI arguments.
