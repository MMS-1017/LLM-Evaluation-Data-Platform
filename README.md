# LLM Evaluation Data Platform

> An end-to-end, production-inspired platform that ingests documents, refines them through a
> Medallion data architecture with **PySpark**, embeds them into a **Qdrant** vector store, serves
> a **RAG** (Retrieval-Augmented Generation) API with **FastAPI**, and continuously evaluates
> answer quality with **Ragas** + **MLflow**.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Spark" src="https://img.shields.io/badge/PySpark-3.5-orange.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-teal.svg">
</p>

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [The data pipeline](#the-data-pipeline)
- [The RAG API](#the-rag-api)
- [Evaluation](#evaluation)
- [Orchestration](#orchestration)
- [Docker](#docker)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## What it does

The platform turns a folder of raw documents into a **queryable, quality-monitored knowledge base**.
It is organized around two ideas that recur throughout modern AI data engineering:

1. **A Medallion (Bronze → Silver → Gold) data architecture.** Raw files are ingested verbatim
   (Bronze), progressively cleaned, chunked, and scored (Silver), then filtered down to a
   high-quality, embedding-ready set (Gold). Each layer is an immutable Parquet dataset, so every
   stage is reproducible and independently debuggable.

2. **A closed evaluation loop.** Once chunks are embedded and indexed, the platform can generate a
   synthetic test set, run a RAG evaluation, and log the metrics to MLflow — so you can detect
   regressions or data drift over time instead of guessing whether a change helped.

Everything is configurable through environment variables and YAML, runs locally with a single
command, and is orchestrated in production with Apache Airflow.

---

## Architecture

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                    DATA PIPELINE                           │
                        │                                                            │
  data/raw/*.txt        │   Ingestion         PySpark            PySpark             │
  .md .json .pdf .csv   │  (Raw→Bronze)     (Bronze→Silver)    (Silver→Gold)         │
        │               │      │                  │                  │              │
        └───────────────┼──►  Bronze  ──────►   Silver   ──────►    Gold            │
                        │   (raw parquet)   (clean + chunk +   (quality-filtered,    │
                        │                    quality score)     dedup'd chunks)      │
                        └───────────────────────────────────────────┬──────────────┘
                                                                     │
                                        ┌────────────────────────────┴───────────────┐
                                        │                                             │
                              Embedding + Indexing                          Evaluation
                              (Gold → Qdrant)                               (Gold → Ragas → MLflow)
                                        │                                             │
                                        ▼                                             ▼
                                 ┌─────────────┐                            ┌──────────────────┐
                                 │   Qdrant    │                            │  MLflow tracking │
                                 │ vector store│                            │ (metrics + report)│
                                 └──────┬──────┘                            └──────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐        POST /query
                              │   FastAPI service │  ◄───────────────────  client
                              │  (RAG retriever)  │  ───────────────────►  grounded answer
                              └───────────────────┘        + sources
```

**RAG request flow (`POST /query`):** embed the question → cosine-similarity search in Qdrant for
the top-*k* chunks → build a context-grounded prompt → generate an answer with the configured LLM →
return the answer together with its source documents and the exact contexts used.

---

## Tech stack

| Area              | Technologies                                                                 |
| ----------------- | ---------------------------------------------------------------------------- |
| Language          | Python 3.11+                                                                 |
| Data engineering  | PySpark 3.5, Delta Lake, Parquet, PyArrow, Pandas                            |
| AI / embeddings   | sentence-transformers (`all-MiniLM-L6-v2`), OpenAI (optional), LangChain     |
| Vector store      | Qdrant (cosine similarity)                                                   |
| Generation (LLM)  | HuggingFace `google/flan-t5-base` (local) or OpenAI                          |
| Serving           | FastAPI + Uvicorn                                                            |
| Evaluation        | Ragas (with a dependency-free heuristic fallback), MLflow                    |
| Orchestration     | Apache Airflow                                                               |
| Config & logging  | Pydantic / pydantic-settings, PyYAML, Loguru                                 |
| Tooling           | pytest, Ruff, Black, MyPy, Docker                                            |

---

## Project structure

```
LLM-Evaluation-Data-Platform/
├── airflow/
│   └── dags/
│       ├── llm_eval_pipeline.py      # Full daily pipeline DAG
│       └── eval_only_pipeline.py     # Weekly evaluation-only DAG
├── configs/
│   ├── pipeline.yaml                 # Paths, chunking, quality filters
│   ├── spark.yaml                    # Spark session tuning
│   └── evaluation.yaml               # Metrics, MLflow, Ragas settings
├── data/
│   └── raw/                          # Sample source documents (6 .txt files)
├── docker/
│   ├── Dockerfile.api                # FastAPI RAG service image
│   ├── Dockerfile.spark              # Spark worker image
│   └── Dockerfile.airflow            # Airflow scheduler/worker image
├── notebooks/
│   └── pipeline_monitoring.ipynb     # Layer-by-layer EDA & metrics dashboard
├── pipelines/
│   ├── ingestion/                    # Raw → Bronze (loaders + writer + state tracker)
│   ├── spark/                        # Bronze → Silver → Gold jobs + session factory
│   ├── embeddings/                   # Embedding service + Qdrant indexer
│   ├── evaluation/                   # Test-set generation, Ragas, MLflow tracking
│   └── orchestration/                # Local end-to-end pipeline runner
├── services/
│   ├── api/                          # FastAPI app, routes, schemas, DI
│   ├── retriever/                    # RAGRetriever (embed → search → generate)
│   └── vectorstore/                  # Typed Qdrant client wrapper
├── src/
│   ├── common/                       # Logging + typed exceptions
│   ├── config/                       # Pydantic settings
│   ├── models/                       # Domain models (documents, evaluation)
│   └── utils/                        # File & text helpers
├── tests/
│   ├── unit/                         # 100+ fast, dependency-free unit tests
│   └── integration/                  # End-to-end tests (no Spark/Qdrant needed)
├── .env.example                      # Copy to .env and edit
├── pyproject.toml                    # Package, dependencies, tooling config
└── plan.md                           # Original design notes
```

---

## Prerequisites

- **Python 3.11 or newer**
- **Java 17** (required by PySpark) — only needed if you run the Spark stages locally
- **A running Qdrant instance** — needed for embedding/indexing and for the `/query` API. The
  quickest way to get one:

  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

- **(Optional) An MLflow tracking server** at `http://localhost:5000` if you want evaluation runs
  persisted to a UI. Evaluation still works without it — MLflow logging is skipped gracefully.
- **(Optional) An OpenAI API key** if you prefer OpenAI over the local HuggingFace models.

---

## Installation

```bash
git clone https://github.com/MMS-1017/LLM-Evaluation-Data-Platform.git
cd LLM-Evaluation-Data-Platform

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[all]"              # everything, including dev tooling
```

Dependencies are grouped into **optional extras** so you can install only what a given task needs:

| Extra      | Installs                                                        | Use when you need…            |
| ---------- | -------------------------------------------------------------- | ----------------------------- |
| `spark`    | PySpark, Delta Lake, PyArrow, Pandas                           | the Bronze/Silver/Gold jobs   |
| `ai`       | LangChain, sentence-transformers, transformers, torch, Qdrant | embeddings + generation       |
| `api`      | FastAPI, Uvicorn, httpx                                        | the RAG service               |
| `eval`     | Ragas, MLflow                                                  | RAG evaluation                |
| `config`   | Pydantic, pydantic-settings, python-dotenv, PyYAML             | settings (always useful)      |
| `logging`  | Loguru                                                         | structured logging            |
| `dev`      | pytest, Ruff, Black, MyPy                                      | contributing                  |
| `all`      | all of the above                                              | one-shot local setup          |

Example of a slimmer install for just serving the API:

```bash
pip install -e ".[ai,api,config,logging]"
```

---

## Configuration

All runtime configuration is loaded from environment variables (and an optional `.env` file) via
Pydantic settings. Start from the template:

```bash
cp .env.example .env
```

Key variables (see [`.env.example`](.env.example) for the full list):

| Variable                 | Default              | Description                                          |
| ------------------------ | -------------------- | ---------------------------------------------------- |
| `EMBEDDING_PROVIDER`     | `huggingface`        | `huggingface` (no key) or `openai`                   |
| `EMBEDDING_MODEL`        | `all-MiniLM-L6-v2`   | Sentence-transformer model name                      |
| `EMBEDDING_DIMENSION`    | `384`                | Must match the embedding model                       |
| `LLM_PROVIDER`           | `huggingface`        | `huggingface` (local) or `openai`                    |
| `LLM_MODEL`              | `google/flan-t5-base`| Generation model                                     |
| `OPENAI_API_KEY`         | *(empty)*            | Required only when a provider is set to `openai`     |
| `QDRANT_HOST` / `_PORT`  | `localhost` / `6333` | Qdrant connection                                    |
| `QDRANT_COLLECTION`      | `llm_eval_docs`      | Collection name                                      |
| `MLFLOW_TRACKING_URI`    | `http://localhost:5000` | MLflow server (optional)                          |
| `CHUNK_SIZE` / `_OVERLAP`| `512` / `64`         | Sliding-window chunking parameters                   |
| `API_TOP_K`              | `5`                  | Default number of chunks retrieved per query         |

> **Note:** the more detailed YAML files in [`configs/`](configs/) document pipeline defaults
> (quality thresholds, Spark tuning, evaluation metrics) and are a useful reference even though the
> runtime is driven primarily by environment variables.

---

## Quickstart

The fastest way to see the whole thing work end-to-end on the bundled sample documents:

```bash
# 1. Start Qdrant (in a separate terminal)
docker run -p 6333:6333 qdrant/qdrant

# 2. Configure
cp .env.example .env

# 3. Run the full pipeline: ingest → Spark → embed/index → evaluate
python -m pipelines.orchestration.pipeline_runner

# 4. Serve the RAG API
python -m services.api.main                 # or: uvicorn services.api.main:app --reload

# 5. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a transformer model?", "top_k": 5}'
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

The pipeline runner accepts flags to run a subset of stages — handy during development:

```bash
python -m pipelines.orchestration.pipeline_runner --skip-spark   # reuse existing Silver/Gold
python -m pipelines.orchestration.pipeline_runner --only-eval    # evaluation only
python -m pipelines.orchestration.pipeline_runner --force-ingest # re-ingest all raw files
```

Each stage can also be run on its own:

```bash
python -m pipelines.ingestion.run_ingestion       # Raw     → Bronze (incremental)
python -m pipelines.ingestion.run_ingestion --force  # force full re-ingest
python -m pipelines.spark.bronze_to_silver         # Bronze  → Silver
python -m pipelines.spark.silver_to_gold           # Silver  → Gold
python -m pipelines.embeddings.run_embeddings      # Gold    → Qdrant
python -m pipelines.evaluation.run_evaluation      # Gold    → Ragas → MLflow
```

---

## The data pipeline

The platform implements a classic **Medallion architecture**. Each arrow below is a stage you can
run independently; each box is an immutable Parquet dataset under `data/`.

### 1. Ingestion — Raw → Bronze (incremental)

`DocumentLoader` reads every supported file (`.txt`, `.md`, `.json`, `.pdf`, `.csv`) from
`data/raw/` into a typed `RawDocument`, computing a SHA-256 `doc_id` for content-addressable
deduplication. `BronzeWriter` serializes each batch to a timestamped Parquet file
(`data/bronze/batch_YYYYMMDD_HHMMSS.parquet`). The Bronze layer stores content **verbatim** — no
transformations, only provenance metadata.

**Incremental ingestion** is the default: `IngestStateTracker` records which `doc_id`s have
already been ingested in a JSON state file (`data/bronze/ingestion_state.json`).  On subsequent
runs only new or changed files are processed, making repeated runs fast and idempotent.  Pass
`--force` (or `--force-ingest` on the orchestration runner) to reset the state and re-ingest
everything.

### 2. Bronze → Silver (PySpark)

A Spark job that:

1. **Deduplicates** by `doc_id`, keeping the most recently ingested copy.
2. **Filters** out documents shorter than 100 characters.
3. **Cleans** text (Unicode normalization, whitespace collapse, boilerplate removal).
4. **Chunks** each document with a sliding window (`CHUNK_SIZE`/`CHUNK_OVERLAP`).
5. **Scores** each chunk's quality (alphabetic ratio + lexical diversity) and estimates a token
   count.

The result is one row per chunk in `data/silver/`.

### 3. Silver → Gold (PySpark)

A second Spark job that keeps only embedding-worthy chunks: quality score ≥ `0.30`, token count
≥ `20`, deduplicated by `chunk_id`, sorted for stable output. The result in `data/gold/` is the
canonical input to the embedding pipeline.

### 4. Embedding + Indexing — Gold → Qdrant

`VectorIndexer` reads the Gold Parquet in configurable batches, embeds each chunk with
`EmbeddingService` (HuggingFace sentence-transformers by default, OpenAI optionally), and upserts
the vectors into Qdrant via a typed `QdrantVectorStore` wrapper. Collection creation is idempotent
(cosine distance), and string `chunk_id`s are mapped to stable unsigned integer point IDs, so the
whole stage is safe to re-run.

> **Data layers at a glance**
>
> | Layer  | Location        | Grain             | Contents                                        |
> | ------ | --------------- | ----------------- | ----------------------------------------------- |
> | Raw    | `data/raw/`     | file              | Original source documents                       |
> | Bronze | `data/bronze/`  | document          | Verbatim content + provenance metadata          |
> | Silver | `data/silver/`  | chunk             | Cleaned, chunked, quality-scored text           |
> | Gold   | `data/gold/`    | chunk (filtered)  | High-quality, deduplicated, embedding-ready set |
>
> The `data/bronze`, `data/silver`, and `data/gold` directories are git-ignored — only the schema
> lives in the repo, not the generated data.

---

## The RAG API

The FastAPI service exposes the platform over HTTP.

| Method & path | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| `GET  /health`| Qdrant connectivity + collection statistics                        |
| `POST /query` | RAG question answering (retrieve top-*k* chunks, then generate)     |
| `POST /ingest`| Trigger document ingestion into the Bronze layer                    |
| `GET  /docs`  | Swagger UI (auto-generated)                                         |
| `GET  /redoc` | ReDoc UI (auto-generated)                                           |

**Example — `POST /query`:**

```json
// Request
{ "question": "What is a transformer model?", "top_k": 5 }

// Response
{
  "question": "What is a transformer model?",
  "answer": "A transformer is a neural network architecture that relies on self-attention...",
  "sources": ["data/raw/transformers.txt"],
  "contexts": ["...retrieved chunk text used to ground the answer..."],
  "top_k": 5
}
```

The retriever degrades gracefully: if no relevant chunks are found it says so, and if LLM
generation fails it falls back to returning the most relevant retrieved context.

Run it locally with:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Evaluation

The evaluation pipeline closes the quality loop in three steps:

1. **`TestSetGenerator`** samples chunks from the Gold layer and builds synthetic
   question/reference-answer pairs using deterministic templates (seeded for reproducibility, no API
   key required).
2. **`RagasEvaluator`** scores each sample on `faithfulness`, `answer_relevancy`,
   `context_precision`, and `context_recall`. If Ragas (and an LLM backend) is available it runs the
   real metrics; otherwise it falls back to a fast, dependency-free **heuristic** scorer — ideal for
   CI where you want a signal without paying for LLM calls.
3. **`MLflowTracker`** logs the run's parameters, aggregate metrics, and the full report (as a JSON
   artifact) to MLflow. If MLflow isn't reachable, the pipeline logs a warning and continues.

```bash
python -m pipelines.evaluation.run_evaluation
```

---

## Orchestration

Two Airflow DAGs live in [`airflow/dags/`](airflow/dags/):

- **`llm_eval_platform_pipeline`** — the full pipeline, scheduled `@daily`, capped at one active run
  to prevent data races:

  ```
  ingest_raw_documents → spark_bronze_to_silver → spark_silver_to_gold → embed_and_index → evaluate_rag
  ```

- **`llm_eval_quality_monitor`** — an `@weekly`, evaluation-only DAG for tracking quality over time
  without re-ingesting, useful for catching model drift or data-quality regressions after
  deployment.

For local runs (development, integration testing, CI) use the plain-Python
[`pipeline_runner`](pipelines/orchestration/pipeline_runner.py) described in
[Quickstart](#quickstart) — it executes the same stages sequentially without an Airflow scheduler.

---

## Docker

Container images are provided for each runtime concern:

```bash
# FastAPI RAG service
docker build -f docker/Dockerfile.api -t llm-eval-api .
docker run -p 8000:8000 --env-file .env llm-eval-api

# Spark worker image
docker build -f docker/Dockerfile.spark -t llm-eval-spark .

# Airflow image (with Java + all pipeline dependencies)
docker build -f docker/Dockerfile.airflow -t llm-eval-airflow .
```

The API image runs as a non-root user and ships with a `HEALTHCHECK` against `/health`.

---

## Development

Common tasks (after `pip install -e ".[dev]"`):

```bash
ruff check .          # lint
black .               # format
mypy src pipelines services   # type-check
pytest                # run the test suite
```

Tooling is configured in [`pyproject.toml`](pyproject.toml): Ruff and Black use a 100-character line
length, and pytest is set up with `unit` and `integration` markers so slow, service-dependent tests
(Qdrant, Spark) can be selected or skipped.

**Coding conventions:** modules use lazy imports for heavy optional dependencies (Spark, torch,
Qdrant) so that lightweight code paths stay importable without them; external failures are wrapped in
typed exceptions from `src/common/exceptions.py`; and all configuration flows through
`src/config/settings.py`.

---

## Roadmap

Items completed since the initial release:

- ✅ `docker-compose.yml` wiring Qdrant, MLflow, and the API with health-checks.
- ✅ Populated `tests/` suite — 165 tests covering unit and integration layers for every
  pipeline stage (ingestion, evaluation, embedding, retrieval, file utils, state tracker).
- ✅ Incremental / streaming ingestion via `IngestStateTracker` — only new or changed
  documents are processed on each run; full re-ingest available with `--force`.
- ✅ `notebooks/pipeline_monitoring.ipynb` — interactive EDA across all Medallion layers
  with data-quality visualisations and a pipeline health-check summary.

Potential future enhancements:

- Streaming ingestion from object storage (S3/GCS) via file-change events.
- Spark Structured Streaming job for near-real-time Bronze → Silver processing.
- Drift detection: automatically flag evaluation runs where metrics drop below a threshold.

---

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
