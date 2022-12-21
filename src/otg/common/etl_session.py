"""Classes to reuse spark connection and logging functionalities."""

from __future__ import annotations

from pyspark.sql import SparkSession


class ETLSession:
    """Spark session class."""

    def __init__(
        self: ETLSession, spark_uri: str, app_name: str, write_mode: str
    ) -> None:
        """Creates spark session and logger."""
        # create session and retrieve Spark logger object
        self.spark = (
            SparkSession.builder.master(spark_uri).appName(app_name).getOrCreate()
        )

        self.logger = Log4j(self.spark)
        self.write_mode = write_mode


class Log4j:
    """Log4j logger class."""

    def __init__(self: Log4j, spark: SparkSession) -> None:
        """Initialise logger.

        Args:
            spark (SparkSession): Available spark session
        """
        # get spark app details with which to prefix all messages
        conf = spark.sparkContext.getConf()
        app_id = conf.get("spark.app.id")
        app_name = conf.get("spark.app.name")

        log4j = spark._jvm.org.apache.log4j

        message_prefix = f"<{app_name}-{app_id}>"
        self.logger = log4j.LogManager.getLogger(message_prefix)

    def error(self: Log4j, message: str) -> None:
        """Log an error.

        Args:
            message (str): Error message to write to log

        Returns:
            _type_: None
        """
        self.logger.error(message)
        return None

    def warn(self: Log4j, message: str) -> None:
        """Log a warning.

        Args:
            message (str): Warning messsage to write to log

        Returns:
            _type_: None
        """
        self.logger.warn(message)
        return None

    def info(self: Log4j, message: str) -> None:
        """Log information.

        Args:
            message (str): Information message to write to log

        Returns:
            _type_: None
        """
        self.logger.info(message)
        return None
