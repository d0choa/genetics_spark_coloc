"""Utilities to perform colocalisation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np
import pyspark.ml.functions as fml
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.sql.types import DoubleType

from otg.dataset.colocalisation import Colocalisation

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

    from otg.dataset.study_locus_overlap import StudyLocusOverlap


class ECaviar:
    """ECaviar-based colocalisation analysis.

    It extends [CAVIAR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5142122/#bib18) framework to explicitly estimate the posterior probability that the same variant is causal in 2 studies while accounting for the uncertainty of LD. eCAVIAR computes the colocalization posterior probability (**CLPP**) by utilizing the marginal posterior probabilities. This framework allows for **multiple variants to be causal** in a single locus.
    """

    @staticmethod
    def _get_clpp(left_pp: Column, right_pp: Column) -> Column:
        """Calculate the colocalisation posterior probability (CLPP).

        If the fact that the same variant is found causal for two studies are independent events,
        CLPP is defined as the product of posterior porbabilities that a variant is causal in both studies.

        Args:
            left_pp (Column): left posterior probability
            right_pp (Column): right posterior probability

        Returns:
            Column: CLPP
        """
        return left_pp * right_pp

    @classmethod
    def colocalise(
        cls: Type[Colocalisation], overlapping_signals: StudyLocusOverlap
    ) -> Colocalisation:
        """Calculate bayesian colocalisation based on overlapping signals.

        Args:
            overlapping_signals (StudyLocusOverlap): overlapping signals.

        Returns:
            Colocalisation: colocalisation results based on eCAVIAR.
        """
        return Colocalisation(
            _df=(
                overlapping_signals.df.withColumn(
                    "clpp",
                    ECaviar._get_clpp(
                        f.col("left_posteriorProbability"),
                        f.col("right_posteriorProbability"),
                    ),
                )
                .groupBy("left_studyLocusId", "right_studyLocusId")
                .agg(
                    f.count("*").alias("coloc_n_vars"),
                    f.sum(f.col("clpp")).alias("clpp"),
                )
                .withColumn("colocalisationMethod", f.lit("eCAVIAR"))
            )
        )


class Coloc:
    """Calculate bayesian colocalisation based on overlapping signals from credible sets.

    Based on the [R COLOC package](https://github.com/chr1swallace/coloc/blob/main/R/claudia.R), which uses the Bayes factors from the credible set to estimate the posterior probability of colocalisation. This method makes the simplifying assumption that **only one single causal variant** exists for any given trait in any genomic region.

    | Hypothesis | Description                                                           |
    | ---------- | --------------------------------------------------------------------- |
    | H_0        | no association with either trait in the region                        |
    | H_1        | association with trait 1 only                                         |
    | H_2        | association with trait 2 only                                         |
    | H_3        | both traits are associated, but have different single causal variants |
    | H_4        | both traits are associated and share the same single causal variant   |

    !!! warning "Approximate Bayes factors required"
        Coloc requires the availability of approximate Bayes factors (ABF) for each variant in the credible set (`logABF` column).

    """

    @staticmethod
    def _get_logsum(log_abf: VectorUDT) -> float:
        """Calculates logsum of vector.

        This function calculates the log of the sum of the exponentiated
        logs taking out the max, i.e. insuring that the sum is not Inf

        Args:
            log_abf (VectorUDT): log approximate bayes factor

        Returns:
            float: logsum

        Example:
            >>> l = [0.2, 0.1, 0.05, 0]
            >>> round(_get_logsum(l), 6)
            1.476557
        """
        themax = np.max(log_abf)
        result = themax + np.log(np.sum(np.exp(log_abf - themax)))
        return float(result)

    @staticmethod
    def _get_posteriors(all_abfs: VectorUDT) -> DenseVector:
        """Calculate posterior probabilities for each hypothesis.

        Args:
            all_abfs (VectorUDT): h0-h4 bayes factors

        Returns:
            DenseVector: Posterior
        """
        diff = all_abfs - Coloc._get_logsum(all_abfs)
        abfs_posteriors = np.exp(diff)
        return Vectors.dense(abfs_posteriors)

    @classmethod
    def colocalise(
        cls: Type[Coloc],
        overlapping_signals: StudyLocusOverlap,
        priorc1: float = 1e-4,
        priorc2: float = 1e-4,
        priorc12: float = 1e-5,
    ) -> DataFrame:
        """Calculate bayesian colocalisation based on overlapping signals.

        Args:
            overlapping_signals (StudyLocusOverlap): overlapping peaks
            priorc1 (float): Prior on variant being causal for trait 1. Defaults to 1e-4.
            priorc2 (float): Prior on variant being causal for trait 2. Defaults to 1e-4.
            priorc12 (float): Prior on variant being causal for traits 1 and 2. Defaults to 1e-5.

        Returns:
            DataFrame: Colocalisation results
        """
        # register udfs
        logsum = f.udf(Coloc._get_logsum, DoubleType())
        posteriors = f.udf(Coloc._get_posteriors, VectorUDT())
        return Colocalisation(
            _df=(
                overlapping_signals
                # Before summing log_abf columns nulls need to be filled with 0:
                .fillna(0, subset=["left_logABF", "right_logABF"])
                # Sum of log_abfs for each pair of signals
                .withColumn("sum_log_abf", f.col("left_logABF") + f.col("right_logABF"))
                # Group by overlapping peak and generating dense vectors of log_abf:
                .groupBy("chromosome", "left_studyLocusId", "right_studyLocusId")
                .agg(
                    f.count("*").alias("coloc_n_vars"),
                    fml.array_to_vector(f.collect_list(f.col("left_logABF"))).alias(
                        "left_logABF"
                    ),
                    fml.array_to_vector(f.collect_list(f.col("right_logABF"))).alias(
                        "right_logABF"
                    ),
                    fml.array_to_vector(f.collect_list(f.col("sum_log_abf"))).alias(
                        "sum_log_abf"
                    ),
                )
                .withColumn("logsum1", logsum(f.col("left_logABF")))
                .withColumn("logsum2", logsum(f.col("right_logABF")))
                .withColumn("logsum12", logsum(f.col("sum_log_abf")))
                .drop("left_logABF", "right_logABF", "sum_log_abf")
                # Add priors
                # priorc1 Prior on variant being causal for trait 1
                .withColumn("priorc1", f.lit(priorc1))
                # priorc2 Prior on variant being causal for trait 2
                .withColumn("priorc2", f.lit(priorc2))
                # priorc12 Prior on variant being causal for traits 1 and 2
                .withColumn("priorc12", f.lit(priorc12))
                # h0-h2
                .withColumn("lH0abf", f.lit(0))
                .withColumn("lH1abf", f.log(f.col("priorc1")) + f.col("logsum1"))
                .withColumn("lH2abf", f.log(f.col("priorc2")) + f.col("logsum2"))
                # h3
                .withColumn("sumlogsum", f.col("logsum1") + f.col("logsum2"))
                # exclude null H3/H4s: due to sumlogsum == logsum12
                .filter(f.col("sumlogsum") != f.col("logsum12"))
                .withColumn("max", f.greatest("sumlogsum", "logsum12"))
                .withColumn(
                    "logdiff",
                    (
                        f.col("max")
                        + f.log(
                            f.exp(f.col("sumlogsum") - f.col("max"))
                            - f.exp(f.col("logsum12") - f.col("max"))
                        )
                    ),
                )
                .withColumn(
                    "lH3abf",
                    f.log(f.col("priorc1"))
                    + f.log(f.col("priorc2"))
                    + f.col("logdiff"),
                )
                .drop("right_logsum", "left_logsum", "sumlogsum", "max", "logdiff")
                # h4
                .withColumn("lH4abf", f.log(f.col("priorc12")) + f.col("logsum12"))
                # cleaning
                .drop(
                    "priorc1", "priorc2", "priorc12", "logsum1", "logsum2", "logsum12"
                )
                # posteriors
                .withColumn(
                    "allABF",
                    fml.array_to_vector(
                        f.array(
                            f.col("lH0abf"),
                            f.col("lH1abf"),
                            f.col("lH2abf"),
                            f.col("lH3abf"),
                            f.col("lH4abf"),
                        )
                    ),
                )
                .withColumn(
                    "posteriors", fml.vector_to_array(posteriors(f.col("allABF")))
                )
                .withColumn("coloc_h0", f.col("posteriors").getItem(0))
                .withColumn("coloc_h1", f.col("posteriors").getItem(1))
                .withColumn("coloc_h2", f.col("posteriors").getItem(2))
                .withColumn("coloc_h3", f.col("posteriors").getItem(3))
                .withColumn("coloc_h4", f.col("posteriors").getItem(4))
                .withColumn("coloc_h4_h3", f.col("coloc_h4") / f.col("coloc_h3"))
                .withColumn("coloc_log2_h4_h3", f.log2(f.col("coloc_h4_h3")))
                # clean up
                .drop(
                    "posteriors",
                    "allABF",
                    "coloc_h4_h3",
                    "lH0abf",
                    "lH1abf",
                    "lH2abf",
                    "lH3abf",
                    "lH4abf",
                )
                .withColumn("colocalisationMethod", f.lit("COLOC"))
            )
        )
