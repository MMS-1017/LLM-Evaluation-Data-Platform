"""
Lightweight Airflow DAG: evaluation-only quality monitoring.

Runs weekly to track RAG pipeline quality over time without re-ingesting data.
Useful for detecting model drift or data quality degradation post-deployment.
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
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

_PROJECT_ROOT = str(Path(__file__).parents[2])


def task_evaluate() -> None:
    import sys
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from pipelines.evaluation.run_evaluation import run
    code = run()
    if code != 0:
        raise RuntimeError("RAG evaluation pipeline failed")


with DAG(
    dag_id="llm_eval_quality_monitor",
    default_args=DEFAULT_ARGS,
    description="Weekly RAG quality monitoring — evaluation only, no re-ingestion",
    schedule_interval="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["llm", "evaluation", "monitoring"],
) as dag:
    evaluate = PythonOperator(
        task_id="evaluate_rag",
        python_callable=task_evaluate,
    )
