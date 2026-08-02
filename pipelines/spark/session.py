"""
SparkSessionFactory — creates a PySpark session configured for Delta Lake.

Designed for local mode (development) with a single call to
configure_spark_with_delta_pip(), which auto-resolves Delta Lake JARs.
For cluster deployment, override spark_master in settings/env.
"""
from __future__ import annotations

from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)


class SparkSessionFactory:
    """
    Creates and configures a PySpark session with Delta Lake support.

    Example:
        factory = SparkSessionFactory()
        spark = factory.create()
        # ... run jobs ...
        SparkSessionFactory.stop(spark)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

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

        log.info(f"Creating Spark session (master={self.settings.spark_master})")

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
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Disable verbose Spark UI for non-interactive use
            .config("spark.ui.showConsoleProgress", "false")
        )

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        log.info("Spark session ready")
        return spark

    @staticmethod
    def stop(spark) -> None:
        """Gracefully stop the given SparkSession."""
        log.info("Stopping Spark session")
        spark.stop()
