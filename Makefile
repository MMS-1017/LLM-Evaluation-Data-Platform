# ══════════════════════════════════════════════════════════════════════════════
# LLM Evaluation Data Platform — Developer Makefile
# ══════════════════════════════════════════════════════════════════════════════
# Usage:
#   make install        Install all dependencies into the active venv
#   make install-dev    Install only core + dev dependencies
#   make test           Run the full test suite
#   make test-unit      Run only fast unit tests (no external services)
#   make test-int       Run integration tests (no Spark/Qdrant needed)
#   make lint           Run Ruff linter
#   make format         Auto-format with Black
#   make typecheck      Run MyPy
#   make check          lint + typecheck (pre-commit gate)
#   make ingest         Run the ingestion pipeline (Raw → Bronze)
#   make spark          Run both Spark jobs (Bronze → Silver → Gold)
#   make embed          Run embedding + indexing (Gold → Qdrant)
#   make evaluate       Run RAG evaluation (Ragas + MLflow)
#   make pipeline       Run the full pipeline end-to-end
#   make api            Start the FastAPI RAG service (reload mode)
#   make up             Start Docker services (Qdrant + MLflow)
#   make down           Stop Docker services
#   make clean          Remove generated artefacts
# ══════════════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
PYTHON        := python
PIP           := pip
PYTEST        := pytest
RUFF          := ruff
BLACK         := black
MYPY          := mypy

# ── Installation ───────────────────────────────────────────────────────────────

.PHONY: install
install:  ## Install all optional extras (spark, ai, api, eval, config, logging, dev)
	$(PIP) install -e ".[all]"

.PHONY: install-dev
install-dev:  ## Install config, logging, api and dev tooling only
	$(PIP) install -e ".[config,logging,api,dev]"

# ── Testing ────────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run the full test suite
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-unit
test-unit:  ## Run only unit tests (fast, no external services)
	$(PYTEST) tests/unit/ -v --tb=short -m unit

.PHONY: test-int
test-int:  ## Run integration tests (no Spark/Qdrant required)
	$(PYTEST) tests/integration/ -v --tb=short -m integration

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ --cov=src --cov=pipelines --cov=services \
	    --cov-report=term-missing --cov-report=html

# ── Code quality ───────────────────────────────────────────────────────────────

.PHONY: lint
lint:  ## Run Ruff linter
	$(RUFF) check src/ pipelines/ services/ tests/

.PHONY: lint-fix
lint-fix:  ## Run Ruff linter and auto-fix
	$(RUFF) check --fix src/ pipelines/ services/ tests/

.PHONY: format
format:  ## Auto-format with Black
	$(BLACK) src/ pipelines/ services/ tests/

.PHONY: format-check
format-check:  ## Check formatting without modifying files
	$(BLACK) --check src/ pipelines/ services/ tests/

.PHONY: typecheck
typecheck:  ## Run MyPy type checker
	$(MYPY) src/ pipelines/ services/

.PHONY: check
check: lint format-check typecheck  ## Run all quality checks (pre-commit gate)

# ── Pipeline stages ────────────────────────────────────────────────────────────

.PHONY: ingest
ingest:  ## Run ingestion pipeline: Raw → Bronze
	$(PYTHON) -m pipelines.ingestion.run_ingestion

.PHONY: spark
spark:  ## Run Spark jobs: Bronze → Silver → Gold (requires Java 17)
	$(PYTHON) -m pipelines.spark.bronze_to_silver
	$(PYTHON) -m pipelines.spark.silver_to_gold

.PHONY: embed
embed:  ## Run embedding + indexing: Gold → Qdrant (requires Qdrant running)
	$(PYTHON) -m pipelines.embeddings.run_embeddings

.PHONY: evaluate
evaluate:  ## Run RAG evaluation: Gold → Ragas → MLflow
	$(PYTHON) -m pipelines.evaluation.run_evaluation

.PHONY: pipeline
pipeline:  ## Run the full pipeline end-to-end
	$(PYTHON) -m pipelines.orchestration.pipeline_runner

.PHONY: pipeline-skip-spark
pipeline-skip-spark:  ## Run pipeline skipping Spark (reuse existing Silver/Gold)
	$(PYTHON) -m pipelines.orchestration.pipeline_runner --skip-spark

.PHONY: eval-only
eval-only:  ## Run evaluation only (skip ingestion, Spark, embedding)
	$(PYTHON) -m pipelines.orchestration.pipeline_runner --only-eval

# ── API service ────────────────────────────────────────────────────────────────

.PHONY: api
api:  ## Start FastAPI RAG service in reload mode
	uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: api-prod
api-prod:  ## Start FastAPI RAG service in production mode
	$(PYTHON) -m services.api.main

# ── Docker ─────────────────────────────────────────────────────────────────────

.PHONY: up
up:  ## Start Qdrant and MLflow via Docker Compose
	docker compose up -d qdrant mlflow

.PHONY: up-all
up-all:  ## Start all services (Qdrant, MLflow, FastAPI)
	docker compose up -d

.PHONY: down
down:  ## Stop all Docker services
	docker compose down

.PHONY: build
build:  ## Build the FastAPI Docker image
	docker build -f docker/Dockerfile.api -t llm-eval-api .

.PHONY: logs
logs:  ## Tail Docker service logs
	docker compose logs -f

# ── Cleanup ────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove generated artefacts (cache, logs, __pycache__, egg-info)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "Clean complete."

.PHONY: clean-data
clean-data:  ## Remove generated data layers (bronze, silver, gold) — keeps raw
	rm -rf data/bronze data/silver data/gold
	mkdir -p data/bronze data/silver data/gold
	@echo "Data layers cleared."

# ── Help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'
