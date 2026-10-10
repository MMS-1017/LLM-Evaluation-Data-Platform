"""
SparkSessionFactory — creates a PySpark session configured for Delta Lake.

Designed for local mode (development) with a single call to
configure_spark_with_delta_pip(), which auto-resolves Delta Lake JARs.
For cluster deployment, override spark_master in settings/env.

Configuration precedence (highest → lowest):
  1. Environment variables / pydantic Settings
  2. configs/spark.yaml
  3. Hard-coded defaults in this module
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)

_SPARK_YAML = Path(__file__).parents[3] / "configs" / "spark.yaml"


def _load_spark_yaml() -> dict[str, Any]:
    """Load configs/spark.yaml if present; return empty dict otherwise."""
    try:
        import yaml  # pyyaml — part of the 'config' optional extra

        if _SPARK_YAML.exists():
            with _SPARK_YAML.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data.get("spark", {})
    except Exception as exc:
        log.debug(f"Could not load spark.yaml: {exc}")
    return {}


class SparkSessionFactory:
    """
    Creates and configures a PySpark session with Delta Lake support.

    Configuration is sourced first from the Pydantic Settings object
    (environment variables / .env file), then from configs/spark.yaml.

    Example:
        factory = SparkSessionFactory()
        spark = factory.create()
        # ... run jobs ...
        SparkSessionFactory.stop(spark)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._yaml_cfg = _load_spark_yaml()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: str) -> str:
        """
        Return config value with precedence: Settings attr → YAML → default.
        Accesses nested YAML keys with dot notation, e.g. 'sql.shuffle_partitions'.
        """
        # Settings (env vars) always win — check against the hard-coded default
        # to detect if the user has actually set the env var.
        settings_map = {
            "master": self.settings.spark_master,
            "app_name": self.settings.spark_app_name,
            "memory.executor": self.settings.spark_executor_memory,
            "memory.driver": self.settings.spark_driver_memory,
        }
        if key in settings_map:
            return settings_map[key]

        # Walk the YAML dict with dot-separated key
        node: Any = self._yaml_cfg
        for part in key.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = None
                break

        return str(node) if node is not None else default

    # ── Public API ─────────────────────────────────────────────────────────────

    def create(self):
        """Build and return a configured SparkSession."""
        try:
            from pyspark.sql import SparkSession
            from delta import configure_spark_with_delta_pip
        except ImportError as exc:
            raise ImportError(
                "PySpark and delta-spark are required.\n"
                "Install with: pip install 'pyspark==3.5.1' 'delta-spark==3.1.0'"
            ) from exc

        shuffle_partitions = self._cfg("sql.shuffle_partitions", "8")
        log_level = self._cfg("log_level", "WARN")

        log.info(f"Creating Spark session (master={self.settings.spark_master})")
        if self._yaml_cfg:
            log.debug(f"Spark YAML overrides loaded from {_SPARK_YAML}")

        builder = (
            SparkSession.builder.appName(self.settings.spark_app_name)
            .master(self.settings.spark_master)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.executor.memory", self.settings.spark_executor_memory)
            .config("spark.driver.memory", self.settings.spark_driver_memory)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Disable verbose Spark UI for non-interactive use
            .config("spark.ui.showConsoleProgress", "false")
        )

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel(log_level)
        log.info("Spark session ready")
        return spark

    @staticmethod
    def stop(spark) -> None:
        """Gracefully stop the given SparkSession."""
        log.info("Stopping Spark session")
        spark.stop()
