# PLAN.md
# LLM Evaluation Data Platform
## End-to-End AI Data Engineering Project

## Objective

Build a production-inspired platform that continuously ingests documents, processes them with PySpark, generates embeddings, indexes them into a vector database, and evaluates a RAG pipeline automatically.

The emphasis is on:

- Data Engineering
- AI Engineering
- Production architecture
- Reliability
- Scalability
- Clean code
- Testability

This project should look like something that could exist inside a real AI company.

---

# Tech Stack

Language

- Python 3.11+

Data Engineering

- PySpark
- Delta Lake
- Apache Airflow
- Parquet

AI

- LangChain
- Cohere/OpenAI/HuggingFace embeddings
- Qdrant

Serving

- FastAPI

Evaluation

- Ragas
- MLflow

Infrastructure

- Docker
- Docker Compose

Utilities

- Pydantic
- Pandas
- PyArrow
- Loguru

Testing

- pytest

Formatting

- Ruff
- Black

---

# Project Structure

```

project/

│

├── airflow/

├── configs/

├── data/

│ ├── bronze/

│ ├── silver/

│ ├── gold/

│ └── raw/

│

├── docker/

├── docs/

├── notebooks/

├── pipelines/

│ ├── ingestion/

│ ├── spark/

│ ├── embeddings/

│ ├── evaluation/

│ └── orchestration/

│

├── services/

│ ├── api/

│ ├── retriever/

│ └── vectorstore/

│

├── src/

│ ├── common/

│ ├── models/

│ ├── utils/

│ └── config/

│

├── tests/

├── docker-compose.yml

├── pyproject.toml

└── README.md
