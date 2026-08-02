"""
MLflowTracker — logs evaluation results to an MLflow experiment.

Logs:
  - Pipeline parameters (model names, chunking settings, top_k)
  - Aggregate metric scores
  - Full EvaluationReport as a JSON artifact
"""
from __future__ import annotations

from pathlib import Path

from src.common.exceptions import EvaluationError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.evaluation import EvaluationReport

log = get_logger(__name__)


class MLflowTracker:
    """
    Logs an EvaluationReport to MLflow.

    Example:
        tracker = MLflowTracker()
        run_id = tracker.log_report(report)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_report(self, report: EvaluationReport) -> str:
        """
        Log *report* to the configured MLflow experiment.

        Returns:
            The MLflow run ID.

        Raises:
            EvaluationError: If MLflow is not installed or logging fails.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise EvaluationError("mlflow not installed. Run: pip install mlflow") from exc

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        run_name = f"rag_eval_{report.run_id[:8]}"
        log.info(f"Logging evaluation report to MLflow (run_name={run_name})")

        try:
            with mlflow.start_run(run_name=run_name) as mlflow_run:
                # Parameters: reproduce the pipeline configuration
                mlflow.log_params({
                    "embedding_provider": self.settings.embedding_provider,
                    "embedding_model": self.settings.embedding_model,
                    "llm_provider": self.settings.llm_provider,
                    "llm_model": self.settings.llm_model,
                    "chunk_size": self.settings.chunk_size,
                    "chunk_overlap": self.settings.chunk_overlap,
                    "top_k": self.settings.api_top_k,
                    "num_samples": report.num_samples,
                    "pipeline_version": report.pipeline_version,
                    "evaluation_mode": report.metadata.get("mode", "ragas"),
                })

                # Metrics
                for metric_name, value in report.aggregate_metrics.items():
                    mlflow.log_metric(metric_name, value)

                # Artifact: full JSON report
                artifact_dir = Path("logs") / "mlflow_artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                report_file = artifact_dir / f"eval_report_{report.run_id[:8]}.json"
                report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                mlflow.log_artifact(str(report_file))

                mlflow_run_id = mlflow_run.info.run_id
                log.info(f"MLflow run logged: {mlflow_run_id}")
                log.info(f"Metrics: {report.aggregate_metrics}")
                return mlflow_run_id

        except Exception as exc:
            raise EvaluationError(
                "MLflow logging failed",
                details={"run_name": run_name, "error": str(exc)},
            ) from exc
