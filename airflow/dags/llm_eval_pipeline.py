"""
Main Airflow DAG: Full LLM Evaluation Data Platform pipeline.

Task dependency graph:
    ingest_raw_documents
           │
    spark_bronze_to_silver
           │
    spark_silver_to_gold
           │
    embed_and_index
           │
    evaluate_rag

Schedule: daily. Max 1 concurrent run to prevent data races.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "ai-engineering",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

_PROJECT_ROOT = str(Path(__file__).parents[2])


def _add_project_to_path() -> None:
    """Ensure the project root is on sys.path inside each Airflow task."""
    import sys
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def task_ingest() -> None:
    _add_project_to_path()
    from pipelines.ingestion.run_ingestion import run
    code = run()
    if code != 0:
        raise RuntimeError("Ingestion pipeline failed")


def task_bronze_to_silver() -> None:
    _add_project_to_path()
    from pipelines.spark.bronze_to_silver import run
    code = run()
    if code != 0:
        raise RuntimeError("Bronze → Silver Spark job failed")


def task_silver_to_gold() -> None:
    _add_project_to_path()
    from pipelines.spark.silver_to_gold import run
    code = run()
    if code != 0:
        raise RuntimeError("Silver → Gold Spark job failed")


def task_embed_and_index() -> None:
    _add_project_to_path()
    from pipelines.embeddings.run_embeddings import run
    code = run()
    if code != 0:
        raise RuntimeError("Embedding and indexing pipeline failed")


def task_evaluate() -> None:
    _add_project_to_path()
    from pipelines.evaluation.run_evaluation import run
    code = run()
    if code != 0:
        raise RuntimeError("RAG evaluation pipeline failed")


with DAG(
    dag_id="llm_eval_platform_pipeline",
    default_args=DEFAULT_ARGS,
    description="LLM Evaluation Data Platform: Ingest → Spark → Embed → Evaluate",
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["llm", "evaluation", "data-engineering", "production"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_raw_documents",
        python_callable=task_ingest,
    )
    bronze_to_silver = PythonOperator(
        task_id="spark_bronze_to_silver",
        python_callable=task_bronze_to_silver,
    )
    silver_to_gold = PythonOperator(
        task_id="spark_silver_to_gold",
        python_callable=task_silver_to_gold,
    )
    embed_and_index = PythonOperator(
        task_id="embed_and_index",
        python_callable=task_embed_and_index,
    )
    evaluate = PythonOperator(
        task_id="evaluate_rag",
        python_callable=task_evaluate,
    )

    ingest >> bronze_to_silver >> silver_to_gold >> embed_and_index >> evaluate
