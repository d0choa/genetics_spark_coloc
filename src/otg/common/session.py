"""Classes to reuse spark connection and logging functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType


class Session:
    """This class provides a Spark session and logger."""

    def __init__(  # noqa: D107
        self: Session,
        spark_uri: str = "local[*]",
        write_mode: str = "errorifexists",
        app_name: str = "otgenetics",
        hail_home: str | None = None,
        start_hail: bool = False,
        extended_spark_conf: dict[str, str] | None = None,
    ) -> None:
        """Initialises spark session and logger.

        Args:
            spark_uri (str): Spark URI. Defaults to "local[*]".
            write_mode (str): Spark write mode. Defaults to "errorifexists".
            app_name (str): Spark application name. Defaults to "otgenetics".
            hail_home (str | None): Path to Hail installation. Defaults to None.
            start_hail (bool): Whether to start Hail. Defaults to False.
            extended_spark_conf (dict[str, str] | None): Extended Spark configuration. Defaults to None.
        """
        merged_conf = self._create_merged_config(
            start_hail, hail_home, extended_spark_conf
        )

        self.spark = (
            SparkSession.builder.config(conf=merged_conf)
            .master(spark_uri)
            .appName(app_name)
            .getOrCreate()
        )
        self.logger = Log4j(self.spark)

        self.write_mode = write_mode

        self.hail_home = hail_home
        self.start_hail = start_hail

    def _default_config(self: Session) -> SparkConf:
        """Default spark configuration.

        Returns:
            SparkConf: Default spark configuration.
        """
        return (
            SparkConf()
            # Dynamic allocation
            .set("spark.dynamicAllocation.enabled", "true")
            .set("spark.dynamicAllocation.minExecutors", "2")
            .set("spark.dynamicAllocation.initialExecutors", "2")
            .set(
                "spark.shuffle.service.enabled", "true"
            )  # required for dynamic allocation
        )

    def _hail_config(
        self: Session, start_hail: bool, hail_home: str | None
    ) -> SparkConf:
        """Returns the Hail specific Spark configuration.

        Args:
            start_hail (bool): Whether to start Hail.
            hail_home (str | None): Path to Hail installation.

        Returns:
            SparkConf: Hail specific Spark configuration.

        Raises:
            ValueError: If Hail home is not specified but Hail is requested.
        """
        if not start_hail:
            return SparkConf()
        if not hail_home:
            raise ValueError("Hail home must be specified to start Hail.")
        return (
            SparkConf()
            .set("spark.jars", f"{hail_home}/backend/hail-all-spark.jar")
            .set(
                "spark.driver.extraClassPath", f"{hail_home}/backend/hail-all-spark.jar"
            )
            .set("spark.executor.extraClassPath", "./hail-all-spark.jar")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator")
        )

    def _create_merged_config(
        self: Session,
        start_hail: bool,
        hail_home: str | None,
        extended_spark_conf: dict[str, str] | None,
    ) -> SparkConf:
        """Merges the default, and optionally the Hail and extended configurations if provided.

        Args:
            start_hail (bool): Whether to start Hail.
            hail_home (str | None): Path to Hail installation. Defaults to None.
            extended_spark_conf (dict[str, str] | None): Extended Spark configuration.

        Returns:
            SparkConf: Merged Spark configuration.
        """
        all_settings = (
            self._default_config().getAll()
            + self._hail_config(start_hail, hail_home).getAll()
        )
        if extended_spark_conf:
            all_settings += list(extended_spark_conf.items())
        return SparkConf().setAll(all_settings)

    def read_parquet(
        self: Session,
        path: str,
        schema: StructType,
        **kwargs: bool | float | int | str | None,
    ) -> DataFrame:
        """Reads parquet dataset with a provided schema.

        Args:
            path (str): parquet dataset path
            schema (StructType): Spark schema
            **kwargs (bool | float | int | str | None): Additional arguments to pass to spark.read.parquet

        Returns:
            DataFrame: Dataframe with provided schema
        """
        return self.spark.read.schema(schema).parquet(path, **kwargs)


class Log4j:
    """Log4j logger class."""

    def __init__(self, spark: SparkSession) -> None:
        """Log4j logger class. This class provides a wrapper around the Log4j logging system.

        Args:
            spark (SparkSession): The Spark session used to access Spark context and Log4j logging.
        """
        # get spark app details with which to prefix all messages
        log4j = spark.sparkContext._jvm.org.apache.log4j  # type: ignore[assignment, unused-ignore]
        self.logger = log4j.Logger.getLogger(__name__)

        log4j_logger = spark.sparkContext._jvm.org.apache.log4j  # type: ignore[assignment, unused-ignore]
        self.logger = log4j_logger.LogManager.getLogger(__name__)

    def error(self: Log4j, message: str) -> None:
        """Log an error.

        Args:
            message (str): Error message to write to log
        """
        self.logger.error(message)
        return None

    def warn(self: Log4j, message: str) -> None:
        """Log a warning.

        Args:
            message (str): Warning messsage to write to log
        """
        self.logger.warn(message)
        return None

    def info(self: Log4j, message: str) -> None:
        """Log information.

        Args:
            message (str): Information message to write to log
        """
        self.logger.info(message)
        return None
