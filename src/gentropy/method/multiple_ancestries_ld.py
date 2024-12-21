"""The combination of multiple ancestries for LD matrix."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.functions import array, map_from_arrays
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from gentropy.common.session import Session
from gentropy.dataset.summary_statistics import SummaryStatistics
from gentropy.datasource.gnomad.ld import GnomADLDMatrix


class multipleAncestriesLD:
    """Basic functons for multiple ancestries combination."""

    @staticmethod
    def combineLD(
        LD: dict[str, np.ndarray],
        m_p: np.ndarray,
        alphas: list[float],
        pops: list[str],
        zero_af_substitution: float = 1e-4,
    ) -> dict[str, Any]:
        """Combine LD matrices from multiple ancestries.

        Args:
            LD (dict[str, np.ndarray]): the dictionarry of LD matricies, with dimensions, the keys are the ancestries.
            m_p (np.ndarray): matrix of SNP alellic frequncies with dimensions [n_nsps, n_pops].
            alphas (list[float]): list of coefficints that are population poroprtion; the sum of alphas should be 1.
            pops (list[str]): list of ancestries in the same order as in alphas and in m_p.
            zero_af_substitution (float, optional): the value to substitute for zero allele frequencies. Defaults to 1e-4.

        Returns:
            dict[str, Any]: A dictionary containing the combined LD matrix, varG matrix, and frequency.

        Examples:
            >>> corr_matrix_1 = np.array([
            ...     [1.0, 0.5, 0.3, 0.2],
            ...     [0.5, 1.0, 0.4, 0.3],
            ...     [0.3, 0.4, 1.0, 0.5],
            ...     [0.2, 0.3, 0.5, 1.0]
            ... ])
            >>> corr_matrix_2 = np.array([
            ...     [1.0, 0.6, 0.4, 0.3],
            ...     [0.6, 1.0, 0.5, 0.4],
            ...     [0.4, 0.5, 1.0, 0.6],
            ...     [0.3, 0.4, 0.6, 1.0]
            ... ])
            >>> corr_matrix_3 = np.array([
            ...     [1.0, 0.3, 0.2, 0.1],
            ...     [0.3, 1.0, 0.4, 0.2],
            ...     [0.2, 0.4, 1.0, 0.5],
            ...     [0.1, 0.2, 0.5, 1.0]
            ... ])
            >>> LD = {"pop1": corr_matrix_1, "pop2": corr_matrix_2, "pop3": corr_matrix_3}
            >>> m_p = np.array([
            ...     [0.1, 0.2, 0.3],
            ...     [0.4, 0.5, 0.6],
            ...     [0.7, 0.8, 0.9],
            ...     [0.1, 0.3, 0.5]
            ... ])
            >>> pops = ["pop1", "pop2", "pop3"]
            >>> alphas = [0.4, 0.2, 0.4]
            >>> result = multipleAncestriesLD.combineLD(LD=LD, m_p=m_p, alphas=alphas, pops=pops)
            >>> expected_result = {
            ...     'LD': np.array([
            ...         [1.        , 0.46271571, 0.33509199, 0.2944123 ],
            ...         [0.46271571, 1.        , 0.45839856, 0.35042965],
            ...         [0.33509199, 0.45839856, 1.        , 0.55294339],
            ...         [0.2944123 , 0.35042965, 0.55294339, 1.        ]
            ...     ]),
            ...     'varg': np.array([0.336, 0.516, 0.336, 0.484]),
            ...     'freq': np.array([0.2, 0.5, 0.8, 0.3])
            ... }
            >>> assert np.allclose(result['LD'], expected_result['LD']), "LD matrices are not close enough"
            >>> assert np.allclose(result['varg'], expected_result['varg']), "varg arrays are not close enough"
            >>> assert np.allclose(result['freq'], expected_result['freq']), "Frequency arrays are not close enough"
        """
        m_p[m_p == 0] = zero_af_substitution
        m_p[m_p == 1] = 1 - zero_af_substitution

        n_pops = len(pops)
        alphas = np.array(alphas)
        varG = 2 * m_p * (1 - m_p)

        EXY = []
        for i in range(n_pops):
            pop = pops[i]
            varg = varG[:, i]
            vv = np.sqrt(np.outer(varg, varg))
            exey = 4 * np.outer(m_p[:, i], m_p[:, i])

            ld_pop = LD[pop]
            ld_pop[ld_pop > 1] = 1
            ld_pop[ld_pop < -1] = -1
            upper_triangle = np.triu(ld_pop)
            ld_pop = (
                upper_triangle + upper_triangle.T - np.diag(upper_triangle.diagonal())
            )
            np.fill_diagonal(ld_pop, 1)

            exy = ld_pop * vv + exey
            EXY.append(exy)

        freq = np.apply_along_axis(lambda x: np.sum(alphas * x), axis=1, arr=m_p)
        EXEY = 4 * np.outer(freq, freq)

        EXY_t = alphas[0] * EXY[0]
        for i in range(1, n_pops):
            EXY_t += alphas[i] * EXY[i]

        covxy = EXY_t - EXEY
        varg = np.diag(covxy)
        new_ld = covxy * np.sqrt(1 / np.outer(varg, varg))
        np.fill_diagonal(new_ld, 1)

        return {"LD": new_ld, "varg": varg, "freq": freq}

    @staticmethod
    def prepare_data_frames(
        session: Session,
        chromosomeInput: str,
        start: int,
        end: int,
        path_to_ld_index: str,
        listOfAncestries: list[str] = ["nfe", "afr"],
        gwas: SummaryStatistics | None = None,
        full_list_of_ancestries: list[str] = [
            "oth",
            "amr",
            "fin",
            "ami",
            "mid",
            "nfe",
            "sas",
            "asj",
            "eas",
            "afr",
        ],
        filter_snps: bool = True,
        maf_threshold: float = 0.2,
        z_threshold: float = 3,
    ) -> dict[str, Any] | None:
        """Extract the AFs and LD matrices for the given chromosome and region and the list of ancestires.

        Args:
            session (Session): The pyspark session object.
            chromosomeInput (str): Chromosome number.
            start (int): Start position of the region.
            end (int): End position of the region.
            path_to_ld_index (str): Path to the LD index file with AFs.
            listOfAncestries (list[str], optional): List of ancestries. Defaults to ["nfe","afr"].
            gwas (SummaryStatistics | None): GWAS summary statistics. Defaults to None.
            full_list_of_ancestries (list[str], optional): List of all ancestries. Defaults to ['oth', 'amr', 'fin', 'ami', 'mid', 'nfe', 'sas', 'asj', 'eas', 'afr'].
            filter_snps (bool, optional): Whether to filter the SNPs based on MAF and Z-score. Defaults to True.
            maf_threshold (float, optional): The threshold for minor allele frequency. Defaults to 0.2.
            z_threshold (float, optional): The threshold for Z-score. Defaults to 3.

        Returns:
            dict[str, Any] | None: A dictionary containing the LD and ref_af.
        """
        # PLEASE DO NOT REMOVE THIS LINE
        pd.DataFrame.iteritems = pd.DataFrame.items

        init_df = GnomADLDMatrix().get_locus_index_boundaries_no_studyLocus(
            chromosomeInput, start, end, major_population=listOfAncestries[0]
        )
        init_df = (
            init_df.withColumn(
                "variantId",
                f.concat(
                    f.lit(chromosomeInput),
                    f.lit("_"),
                    f.col("`locus.position`"),
                    f.lit("_"),
                    f.col("alleles")[0],
                    f.lit("_"),
                    f.col("alleles")[1],
                ),
            )
            .select("variantId")
            .distinct()
        )

        for ancestry in listOfAncestries[1:]:
            df = GnomADLDMatrix().get_locus_index_boundaries_no_studyLocus(
                chromosomeInput, start, end, major_population=ancestry
            )
            df = (
                df.withColumn(
                    "variantId",
                    f.concat(
                        f.lit(chromosomeInput),
                        f.lit("_"),
                        f.col("`locus.position`"),
                        f.lit("_"),
                        f.col("alleles")[0],
                        f.lit("_"),
                        f.col("alleles")[1],
                    ),
                )
                .select("variantId")
                .distinct()
            )
            init_df = (
                init_df.join(df, init_df["variantId"] == df["variantId"], how="inner")
                .drop(df["variantId"])
                .distinct()
            )

        ldIndex = session.spark.read.parquet(path_to_ld_index)
        df = ldIndex.withColumn(
            "keys",
            array(
                [
                    f.col("alleleFrequencies")[i]["populationName"]
                    for i in range(len(full_list_of_ancestries))
                ]
            ),
        )
        df = df.withColumn(
            "values",
            array(
                [
                    f.col("alleleFrequencies")[i]["alleleFrequency"]
                    for i in range(len(full_list_of_ancestries))
                ]
            ),
        )
        df = df.withColumn("alleleFrequencies", map_from_arrays(df.keys, df.values))
        ldIndex = df.select(
            "*",
            *[f.col("alleleFrequencies")[key].alias(key) for key in listOfAncestries],
        ).select(["variantId"] + listOfAncestries)

        ref_af = (
            init_df.join(
                ldIndex, init_df["variantId"] == ldIndex["variantId"], how="inner"
            )
            .drop(ldIndex["variantId"])
            .distinct()
        )

        if gwas is not None:
            gwas = gwas.sanity_filter()
            gwas_df = (
                gwas.df.withColumn("z", f.col("beta") / f.col("standardError"))
                .filter(f.col("z").isNotNull())
                .filter(f.col("chromosome") == chromosomeInput)
                .filter(f.col("position") >= start)
                .filter(f.col("position") <= end)
                .select("variantId", "z")
            )
            ref_af = (
                ref_af.join(
                    gwas_df, ref_af["variantId"] == gwas_df["variantId"], how="inner"
                )
                .drop(gwas_df["variantId"])
                .distinct()
            )

        ref_af = ref_af.toPandas()
        ref_af = ref_af.drop_duplicates(subset="variantId")

        if filter_snps:
            m_p = ref_af[listOfAncestries]
            m_p = np.array(m_p)
            Z = np.array(ref_af["z"].tolist())

            mask = (m_p >= maf_threshold) & (m_p <= 1 - maf_threshold)
            rows = np.all(mask, axis=1)
            rows = (np.abs(Z) <= z_threshold) & rows
            ind = np.where(rows)[0]
            if len(ind) < 20:
                logging.warning("Not enough snps in the region.")
                return None

            ref_af = ref_af.iloc[ind, :]

        LD = {}
        for ancestry in listOfAncestries:
            df = GnomADLDMatrix().get_locus_index_boundaries_no_studyLocus(
                chromosomeInput, start, end, major_population=ancestry
            )
            df = df.withColumn(
                "variantId",
                f.concat(
                    f.lit(chromosomeInput),
                    f.lit("_"),
                    f.col("`locus.position`"),
                    f.lit("_"),
                    f.col("alleles")[0],
                    f.lit("_"),
                    f.col("alleles")[1],
                ),
            )
            schema_df = df.schema
            df = df.toPandas()
            df = df.drop_duplicates(subset="variantId")
            matched_df = pd.merge(ref_af, df, on="variantId", how="inner")
            df = matched_df[df.columns]
            df = df.sort_values(by="idx")
            df_spark = session.spark.createDataFrame(df, schema=schema_df)
            gnomad_ld = GnomADLDMatrix.get_numpy_matrix(
                df_spark, gnomad_ancestry=ancestry
            )

            df["index_column"] = df.reset_index(drop=True).index
            df_ordered = ref_af[["variantId"]].merge(df, on="variantId", how="left")
            index = df_ordered["index_column"].to_list()
            gnomad_ld = gnomad_ld[index, :][:, index]

            LD[ancestry] = gnomad_ld

        return {"LD": LD, "ref_af": ref_af}

    @staticmethod
    def fun_for_opt(
        alphas: np.ndarray,
        Z: np.ndarray,
        LD: dict[str, np.ndarray],
        m_p: np.ndarray,
        pops: list[str],
        ind: np.ndarray,
        zero_af_substitution: float = 1e-4,
    ) -> float:
        """Function to optimize the alphas.

        Args:
            alphas (np.ndarray): The coefficients to optimize. The same order as m_p.
            Z (np.ndarray): The Z-scores.
            LD (dict[str, np.ndarray]): The dictionary of LD matrices.
            m_p (np.ndarray): The matrix of SNP alellic frequncies.
            pops (list[str]): The list of ancestries to use. The same order as in m_p.
            ind (np.ndarray): The indices to use.
            zero_af_substitution (float, optional): The value to substitute for zero allele frequencies. Defaults to 1e-4.

        Returns:
            float: The negative log likelihood of the model.
        """
        alphas[alphas < 0] = 1  # Ensure alphas are non-negative
        alphas = alphas / np.sum(alphas)  # Normalize alphas to sum to 1
        new_ld = multipleAncestriesLD.combineLD(
            LD=LD,
            m_p=m_p,
            alphas=alphas,
            pops=pops,
            zero_af_substitution=zero_af_substitution,
        )
        log_likelihood = multivariate_normal.logpdf(
            Z[ind],
            mean=np.zeros(len(Z[ind])),
            cov=new_ld["LD"][ind, :][:, ind],
            allow_singular=True,
        )

        return -log_likelihood

    @staticmethod
    def infer_ancestry_proportions(
        session: Session,
        chromosomeInput: str,
        start: int,
        end: int,
        path_to_ld_index: str,
        gwas: SummaryStatistics,
        listOfAncestries: list[str] = ["nfe", "afr"],
        full_list_of_ancestries: list[str] = [
            "oth",
            "amr",
            "fin",
            "ami",
            "mid",
            "nfe",
            "sas",
            "asj",
            "eas",
            "afr",
        ],
        maf_threshold: float = 0.2,
        number_of_snps: int = 20,
        zero_af_substitution: float = 1e-4,
        z_threshold: float = 3,
        number_of_runs: int = 10,
    ) -> pd.DataFrame | None:
        """Infer the ancestry proportions from the Z-scores and LD matrix.

        Args:
            session (Session): The pyspark session object.
            chromosomeInput (str): Chromosome number.
            start (int): Start position of the region.
            end (int): End position of the region.
            path_to_ld_index (str): Path to the LD index file with AFs.
            gwas (SummaryStatistics): GWAS summary statistics.
            listOfAncestries (list[str], optional): List of ancestries to select from. Defaults to ["nfe", "afr"].
            full_list_of_ancestries (list[str], optional): List of all ancestries. Defaults to ['oth', 'amr', 'fin', 'ami', 'mid', 'nfe', 'sas', 'asj', 'eas', 'afr'].
            maf_threshold (float, optional): The threshold for minor allele frequency. Defaults to 0.2.
            number_of_snps (int, optional): The number of SNPs to use for optimization. Defaults to 20.
            zero_af_substitution (float, optional): The value to substitute for zero allele frequencies. Defaults to 1e-4.
            z_threshold (float, optional): The threshold for Z-score. Defaults to 3.
            number_of_runs (int, optional): The number of runs for optimization. Defaults to 10.

        Returns:
            pd.DataFrame | None: The result of the optimization.
        """
        out_dict = multipleAncestriesLD.prepare_data_frames(
            session=session,
            listOfAncestries=listOfAncestries,
            chromosomeInput=chromosomeInput,
            start=start,
            end=end,
            path_to_ld_index=path_to_ld_index,
            gwas=gwas,
            full_list_of_ancestries=full_list_of_ancestries,
            maf_threshold=maf_threshold,
            z_threshold=z_threshold,
            filter_snps=True,
        )
        if out_dict is not None:
            m_p = out_dict["ref_af"][listOfAncestries]
            m_p = np.array(m_p)
            Z = np.array(out_dict["ref_af"]["z"].tolist())

            mask = (m_p >= maf_threshold) & (m_p <= 1 - maf_threshold)
            rows = np.all(mask, axis=1)
            rows = (np.abs(Z) <= z_threshold) & rows
            ind = np.where(rows)[0]
            if len(ind) < number_of_snps:
                logging.warning("Not enough snps in the region.")
                return None

            initial_alphas = np.ones(len(listOfAncestries))
            bounds = [(0, 1)] * len(listOfAncestries)  # Non-negative bounds for alphas

            out: list[np.ndarray] = []
            for _ in range(number_of_runs):
                ind_to_run = np.random.choice(ind, number_of_snps, replace=False)
                values = out_dict["LD"][listOfAncestries[0]][ind_to_run, :][
                    :, ind_to_run
                ]
                np.fill_diagonal(values, 0)
                avg_squared = np.mean(np.square(values))
                np.fill_diagonal(values, 1)
                if avg_squared >= 0.1:
                    result = minimize(
                        multipleAncestriesLD.fun_for_opt,
                        initial_alphas,
                        args=(
                            Z,
                            out_dict["LD"],
                            m_p,
                            listOfAncestries,
                            ind_to_run,
                            zero_af_substitution,
                        ),
                        bounds=bounds,
                    )
                    out = [result.x / np.sum(result.x)] + out

            transposed_arrays = np.transpose(out)
            means = np.mean(transposed_arrays, axis=1)
            medians = np.median(transposed_arrays, axis=1)
            ses = np.sqrt(np.var(transposed_arrays, axis=1))
            df = pd.DataFrame(
                {
                    "Ancestry": listOfAncestries,
                    "means": means,
                    "medians": medians,
                    "ses": ses,
                }
            )
            return df
        else:
            return None

    @staticmethod
    def run_few_loci(
        session: Session,
        gwas: SummaryStatistics,
        path_to_ld_index: str,
        listOfAncestries: list[str] = ["nfe", "afr"],
        full_list_of_ancestries: list[str] = [
            "oth",
            "amr",
            "fin",
            "ami",
            "mid",
            "nfe",
            "sas",
            "asj",
            "eas",
            "afr",
        ],
        maf_threshold: float = 0.2,
        number_of_snps: int = 20,
        zero_af_substitution: float = 1e-4,
        z_threshold: float = 3,
        number_of_runs: int = 10,
    ) -> list[Any]:
        """Infer the ancestry proportions from the Z-scores and LD matrix.

        Args:
            session (Session): The pyspark session object.
            gwas (SummaryStatistics): GWAS summary statistics.
            path_to_ld_index (str): Path to the LD index file with AFs.
            listOfAncestries (list[str], optional): List of ancestries to select from. Defaults to ["nfe", "afr"].
            full_list_of_ancestries (list[str], optional): List of all ancestries. Defaults to ['oth', 'amr', 'fin', 'ami', 'mid', 'nfe', 'sas', 'asj', 'eas', 'afr'].
            maf_threshold (float, optional): The threshold for minor allele frequency. Defaults to 0.2.
            number_of_snps (int, optional): The number of SNPs to use for optimization. Defaults to 20.
            zero_af_substitution (float, optional): The value to substitute for zero allele frequencies. Defaults to 1e-4.
            z_threshold (float, optional): The threshold for Z-score. Defaults to 3.
            number_of_runs (int, optional): The number of runs for optimization. Defaults to 10.

        Returns:
            list[Any]: The list of ancestry proportions for the given study.
        """
        list_of_loci = [
            ("11", 61_692_980, 61_867_354),
            ("14", 45_035_930, 45_200_890),
        ]
        out: list[np.ndarray] = []
        for locus in list_of_loci:
            chromosomeInput, start, end = locus
            x = multipleAncestriesLD.infer_ancestry_proportions(
                session=session,
                chromosomeInput=chromosomeInput,
                start=start,
                end=end,
                path_to_ld_index=path_to_ld_index,
                gwas=gwas,
                listOfAncestries=listOfAncestries,
                full_list_of_ancestries=full_list_of_ancestries,
                maf_threshold=maf_threshold,
                number_of_snps=number_of_snps,
                zero_af_substitution=zero_af_substitution,
                z_threshold=z_threshold,
                number_of_runs=number_of_runs,
            )
            if x is not None:
                out = [x] + out
        return out
