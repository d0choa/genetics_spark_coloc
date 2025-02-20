"""Unit test configuration."""

from __future__ import annotations

from pathlib import Path

import dbldatagen as dg
import hail as hl
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from gentropy.common.Liftover import LiftOverSpark
from gentropy.common.session import Session
from gentropy.dataset.biosample_index import BiosampleIndex
from gentropy.dataset.colocalisation import Colocalisation
from gentropy.dataset.intervals import Intervals
from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.l2g_gold_standard import L2GGoldStandard
from gentropy.dataset.l2g_prediction import L2GPrediction
from gentropy.dataset.ld_index import LDIndex
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.study_locus_overlap import StudyLocusOverlap
from gentropy.dataset.summary_statistics import SummaryStatistics
from gentropy.dataset.target_index import TargetIndex
from gentropy.dataset.variant_index import VariantIndex
from gentropy.datasource.eqtl_catalogue.finemapping import EqtlCatalogueFinemapping
from gentropy.datasource.eqtl_catalogue.study_index import EqtlCatalogueStudyIndex
from gentropy.datasource.gwas_catalog.associations import StudyLocusGWASCatalog
from gentropy.datasource.gwas_catalog.study_index import StudyIndexGWASCatalog
from utils.spark import get_spark_testing_conf


@pytest.fixture(scope="session", autouse=True)
def spark(tmp_path_factory: pytest.TempPathFactory) -> SparkSession:
    """Local spark session for testing purposes.

    Args:
        tmp_path_factory (pytest.TempPathFactory): pytest fixture

    Returns:
        SparkSession: local spark session
    """
    return (
        SparkSession.builder.config(conf=get_spark_testing_conf())
        .master("local[1]")
        .appName("test")
        .getOrCreate()
    )


@pytest.fixture()
def session() -> Session:
    """Return gentropy Session object."""
    return Session()


@pytest.fixture()
def hail_home() -> str:
    """Return the path to the Hail home directory."""
    return Path(hl.__file__).parent.as_posix()


@pytest.fixture()
def mock_colocalisation(spark: SparkSession) -> Colocalisation:
    """Mock colocalisation dataset."""
    coloc_schema = Colocalisation.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(coloc_schema)
        .withColumnSpec(
            "leftStudyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "rightStudyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec("h0", percentNulls=0.1)
        .withColumnSpec("h1", percentNulls=0.1)
        .withColumnSpec("h2", percentNulls=0.1)
        .withColumnSpec("h3", percentNulls=0.1)
        .withColumnSpec("h4", percentNulls=0.1)
        .withColumnSpec("clpp", percentNulls=0.1)
        .withColumnSpec(
            "colocalisationMethod",
            percentNulls=0.0,
            values=["COLOC", "eCAVIAR"],
        )
    )
    return Colocalisation(_df=data_spec.build(), _schema=coloc_schema)


def mock_study_index_data(spark: SparkSession) -> DataFrame:
    """Mock study index dataset."""
    si_schema = StudyIndex.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(si_schema)
        .withColumnSpec(
            "studyId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "traitFromSourceMappedIds",
            expr="array(cast(rand() AS string))",
            percentNulls=0.1,
        )
        .withColumnSpec(
            "backgroundTraitFromSourceMappedIds",
            expr="array(cast(rand() AS string))",
            percentNulls=0.1,
        )
        .withColumnSpec(
            "discoverySamples",
            expr='array(named_struct("sampleSize", cast(rand() as string), "ancestry", cast(rand() as string)))',
            percentNulls=0.1,
        )
        .withColumnSpec(
            "replicationSamples",
            expr='array(named_struct("sampleSize", cast(rand() as string), "ancestry", cast(rand() as string)))',
            percentNulls=0.1,
        )
        .withColumnSpec(
            "geneId",
            expr="cast(id as string)",
        )
        .withColumnSpec("pubmedId", percentNulls=0.1)
        .withColumnSpec("publicationFirstAuthor", percentNulls=0.1)
        .withColumnSpec("publicationDate", percentNulls=0.1)
        .withColumnSpec("publicationJournal", percentNulls=0.1)
        .withColumnSpec("publicationTitle", percentNulls=0.1)
        .withColumnSpec("initialSampleSize", percentNulls=0.1)
        .withColumnSpec("nCases", percentNulls=0.1)
        .withColumnSpec("nControls", percentNulls=0.1)
        .withColumnSpec("nSamples", percentNulls=0.1)
        .withColumnSpec("summarystatsLocation", percentNulls=0.1)
        .withColumnSpec("studyType", percentNulls=0.0, values=StudyIndex.VALID_TYPES)
    )
    return data_spec.build()


@pytest.fixture()
def mock_study_index(spark: SparkSession) -> StudyIndex:
    """Mock StudyIndex dataset."""
    return StudyIndex(
        _df=mock_study_index_data(spark),
        _schema=StudyIndex.get_schema(),
    )


@pytest.fixture()
def mock_study_index_gwas_catalog(spark: SparkSession) -> StudyIndexGWASCatalog:
    """Mock GWASCatalogStudyIndex dataset."""
    return StudyIndexGWASCatalog(
        _df=mock_study_index_data(spark),
        _schema=StudyIndexGWASCatalog.get_schema(),
    )


@pytest.fixture()
def mock_study_locus_overlap(spark: SparkSession) -> StudyLocusOverlap:
    """Mock StudyLocusOverlap dataset."""
    overlap_schema = StudyLocusOverlap.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(overlap_schema)
        .withColumnSpec(
            "leftStudyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "rightStudyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "tagVariantId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "rightStudyType", percentNulls=0.0, values=StudyIndex.VALID_TYPES
        )
    )

    return StudyLocusOverlap(_df=data_spec.build(), _schema=overlap_schema)


def mock_study_locus_data(spark: SparkSession) -> DataFrame:
    """Mock study_locus dataset."""
    sl_schema = StudyLocus.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(sl_schema)
        .withColumnSpec(
            "variantId",
            expr="cast(id as string)",
        )
        .withColumnSpec("chromosome", percentNulls=0.1)
        .withColumnSpec("position", minValue=100, percentNulls=0.1)
        .withColumnSpec("beta", percentNulls=0.1)
        .withColumnSpec("effectAlleleFrequencyFromSource", percentNulls=0.1)
        .withColumnSpec("standardError", percentNulls=0.1)
        .withColumnSpec("subStudyDescription", percentNulls=0.1)
        .withColumnSpec("pValueMantissa", minValue=1, percentNulls=0.1)
        .withColumnSpec("pValueExponent", minValue=-10, percentNulls=0.1)
        .withColumnSpec(
            "qualityControls",
            expr="array(cast(rand() as string))",
            percentNulls=0.1,
        )
        .withColumnSpec("finemappingMethod", percentNulls=0.1)
        .withColumnSpec(
            "locus",
            expr='array(named_struct("is95CredibleSet", cast(rand() > 0.5 as boolean), "is99CredibleSet", cast(rand() > 0.5 as boolean), "logBF", rand(), "posteriorProbability", rand(), "variantId", cast(floor(rand() * 400) + 1 as string), "beta", rand(), "standardError", rand(), "r2Overall", rand(), "pValueMantissa", rand(), "pValueExponent", rand()))',
            percentNulls=0.1,
        )
    )
    return data_spec.build()


@pytest.fixture()
def mock_study_locus(spark: SparkSession) -> StudyLocus:
    """Mock study_locus dataset."""
    return StudyLocus(
        _df=mock_study_locus_data(spark),
        _schema=StudyLocus.get_schema(),
    )


@pytest.fixture()
def mock_study_locus_gwas_catalog(spark: SparkSession) -> StudyLocusGWASCatalog:
    """Mock study_locus dataset."""
    return StudyLocusGWASCatalog(
        _df=mock_study_locus_data(spark),
        _schema=StudyLocusGWASCatalog.get_schema(),
    )


@pytest.fixture()
def mock_intervals(spark: SparkSession) -> Intervals:
    """Mock intervals dataset."""
    interval_schema = Intervals.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(interval_schema)
        .withColumnSpec("pmid", percentNulls=0.1)
        .withColumnSpec("resourceScore", percentNulls=0.1)
        .withColumnSpec("score", percentNulls=0.1)
        .withColumnSpec("biofeature", percentNulls=0.1)
    )

    return Intervals(_df=data_spec.build(), _schema=interval_schema)


@pytest.fixture()
def mock_variant_index(spark: SparkSession) -> VariantIndex:
    """Mock variant index."""
    vi_schema = VariantIndex.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(vi_schema)
        .withColumnSpec(
            "variantId",
            expr="cast(id as string)",
        )
        .withColumnSpec("mostSevereConsequenceId", percentNulls=0.1)
        # Nested column handling workaround
        # https://github.com/databrickslabs/dbldatagen/issues/135
        # It's a workaround for nested column handling in dbldatagen.
        .withColumnSpec(
            "variantEffect",
            expr="""
                array(
                    named_struct(
                        "method", cast(rand() as string),
                        "assessment", cast(rand() as string),
                        "score", rand(),
                        "assessmentFlag", cast(rand() as string),
                        "targetId", cast(floor(rand() * 400) + 1 as string),
                        "normalizedScore", cast(rand() as float)
                    )
                )
            """,
            percentNulls=0.1,
        )
        .withColumnSpec(
            "alleleFrequencies",
            expr='array(named_struct("alleleFrequency", rand(), "populationName", cast(rand() as string)))',
            percentNulls=0.1,
        )
        .withColumnSpec("rsIds", expr="array(cast(rand() AS string))", percentNulls=0.1)
        .withColumnSpec(
            "transcriptConsequences",
            expr="""
                array(
                    named_struct(
                        "variantFunctionalConsequenceIds", array(cast(rand() as string)),
                        "aminoAcidChange", cast(rand() as string),
                        "uniprotAccessions", array(cast(rand() as string)),
                        "isEnsemblCanonical", cast(rand() as boolean),
                        "codons", cast(rand() as string),
                        "distanceFromTss", cast(floor(rand() * 500000) as long),
                        "distanceFromFootprint", cast(floor(rand() * 500000) as long),
                        "appris", cast(rand() as string),
                        "maneSelect", cast(rand() as string),
                        "targetId", cast(floor(rand() * 400) + 1 as string),
                        "impact", cast(rand() as string),
                        "lofteePrediction", cast(rand() as string),
                        "siftPrediction", rand(),
                        "polyphenPrediction", rand(),
                        "consequenceScore", cast(rand() as float),
                        "transcriptIndex", cast(rand() as integer),
                        "transcriptId", cast(rand() as string),
                        "biotype", 'protein_coding',
                        "approvedSymbol", cast(rand() as string)
                    )
                )
            """,
            percentNulls=0.1,
        )
        .withColumnSpec(
            "dbXrefs",
            expr="""
                array(
                    named_struct(
                        "id", cast(rand() as string),
                        "source", cast(rand() as string)
                    )
                )
            """,
            percentNulls=0.1,
        )
    )

    return VariantIndex(_df=data_spec.build(), _schema=vi_schema)


@pytest.fixture()
def mock_summary_statistics_data(spark: SparkSession) -> DataFrame:
    """Generating mock summary statistics data.

    Args:
        spark (SparkSession): Spark session

    Returns:
        DataFrame: Mock summary statistics data
    """
    ss_schema = SummaryStatistics.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
            name="summaryStats",
        )
        .withSchema(ss_schema)
        .withColumnSpec(
            "studyId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "variantId",
            expr="cast(id as string)",
        )
        # Allowing missingness in effect allele frequency and enforce upper limit:
        .withColumnSpec(
            "effectAlleleFrequencyFromSource", percentNulls=0.1, maxValue=1.0
        )
        # Allowing missingness:
        .withColumnSpec("standardError", percentNulls=0.1)
    )

    return data_spec.build()


@pytest.fixture()
def mock_summary_statistics(
    mock_summary_statistics_data: DataFrame,
) -> SummaryStatistics:
    """Generating a mock summary statistics dataset."""
    return SummaryStatistics(
        _df=mock_summary_statistics_data, _schema=SummaryStatistics.get_schema()
    )


@pytest.fixture()
def mock_ld_index(spark: SparkSession) -> LDIndex:
    """Mock ld index."""
    ld_schema = LDIndex.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(ld_schema)
        .withColumnSpec(
            "variantId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "ldSet",
            expr="array(named_struct('tagVariantId', cast(rand() as string), 'rValues', array(named_struct('population', cast(rand() as string), 'r', cast(rand() as double)))))",
        )
    )

    return LDIndex(_df=data_spec.build(), _schema=ld_schema)


@pytest.fixture()
def sample_gwas_catalog_studies(spark: SparkSession) -> DataFrame:
    """Sample GWAS Catalog studies."""
    return spark.read.csv(
        "tests/gentropy/data_samples/gwas_catalog_studies.tsv",
        sep="\t",
        header=True,
    )


@pytest.fixture()
def sample_gwas_catalog_ancestries_lut(spark: SparkSession) -> DataFrame:
    """Sample GWAS ancestries sample data."""
    return spark.read.csv(
        "tests/gentropy/data_samples/gwas_catalog_ancestries.tsv",
        sep="\t",
        header=True,
    )


@pytest.fixture()
def sample_gwas_catalog_harmonised_sumstats_list(spark: SparkSession) -> DataFrame:
    """Sample GWAS harmonised sumstats sample data."""
    return spark.read.csv(
        "tests/gentropy/data_samples/gwas_catalog_harmonised_list.txt",
        sep="\t",
        header=False,
    )


@pytest.fixture()
def sample_gwas_catalog_associations(spark: SparkSession) -> DataFrame:
    """Sample GWAS raw associations sample data."""
    return spark.read.csv(
        "tests/gentropy/data_samples/gwas_catalog_associations.tsv",
        sep="\t",
        header=True,
    )


@pytest.fixture()
def sample_summary_statistics(spark: SparkSession) -> SummaryStatistics:
    """Sample GWAS raw associations sample data."""
    return SummaryStatistics(
        _df=spark.read.parquet("tests/gentropy/data_samples/sumstats_sample"),
        _schema=SummaryStatistics.get_schema(),
    )


@pytest.fixture()
def sample_finngen_studies(spark: SparkSession) -> DataFrame:
    """Sample FinnGen studies."""
    # For reference, the sample file was generated with the following command:
    # curl https://r9.finngen.fi/api/phenos | jq '.[:10]' > tests/gentropy/data_samples/finngen_studies_sample.json
    with open(
        "tests/gentropy/data_samples/finngen_studies_sample.json"
    ) as finngen_studies:
        json_data = finngen_studies.read()
        rdd = spark.sparkContext.parallelize([json_data])
        return spark.read.json(rdd)


@pytest.fixture()
def sample_eqtl_catalogue_finemapping_credible_sets(session: Session) -> DataFrame:
    """Sample raw eQTL Catalogue credible sets outputted by SuSIE."""
    return EqtlCatalogueFinemapping.read_credible_set_from_source(
        session,
        credible_set_path=["tests/gentropy/data_samples/QTD000584.credible_sets.tsv"],
    )


@pytest.fixture()
def sample_eqtl_catalogue_finemapping_lbf(session: Session) -> DataFrame:
    """Sample raw eQTL Catalogue table with logBayesFactors outputted by SuSIE."""
    return EqtlCatalogueFinemapping.read_lbf_from_source(
        session,
        lbf_path=["tests/gentropy/data_samples/QTD000584.lbf_variable.txt"],
    )


@pytest.fixture()
def sample_eqtl_catalogue_studies_metadata(spark: SparkSession) -> DataFrame:
    """Sample raw eQTL Catalogue table with metadata about the QTD000584 study."""
    return spark.read.option("delimiter", "\t").csv(
        "tests/gentropy/data_samples/sample_eqtl_catalogue_studies.tsv",
        header=True,
        schema=EqtlCatalogueStudyIndex.raw_studies_metadata_schema,
    )


@pytest.fixture()
def sample_ukbiobank_studies(spark: SparkSession) -> DataFrame:
    """Sample UKBiobank manifest."""
    # Sampled 10 rows of the UKBB manifest tsv
    return spark.read.csv(
        "tests/gentropy/data_samples/neale2_saige_study_manifest.samples.tsv",
        sep="\t",
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def study_locus_sample_for_colocalisation(spark: SparkSession) -> DataFrame:
    """Sample study locus data for colocalisation."""
    return StudyLocus(
        _df=spark.read.parquet("tests/gentropy/data_samples/coloc_test.parquet"),
        _schema=StudyLocus.get_schema(),
    )


@pytest.fixture()
def sample_target_index(spark: SparkSession) -> DataFrame:
    """Sample target index sample data."""
    return spark.read.parquet(
        "tests/gentropy/data_samples/target_sample.parquet",
    )


@pytest.fixture()
def mock_target_index(spark: SparkSession) -> TargetIndex:
    """Mock target index dataset."""
    ti_schema = TargetIndex.get_schema()

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=30,
            partitions=4,
            randomSeedMethod="hash_fieldname",
            seedColumnName="_id",  # required as the target_index has the id column
        )
        .withSchema(ti_schema)
        .withColumnSpec(
            "id",
            expr="cast(_id as string)",
        )
        .withColumnSpec("approvedSymbol", percentNulls=0.1)
        .withColumnSpec(
            "biotype", percentNulls=0.1, values=["protein_coding", "lncRNA"]
        )
        .withColumnSpec("approvedName", percentNulls=0.1)
        .withColumnSpec("tss", percentNulls=0.1)
        .withColumnSpec("genomicLocation", percentNulls=0.1)
    )

    return TargetIndex(_df=data_spec.build(), _schema=ti_schema)


@pytest.fixture()
def mock_biosample_index(spark: SparkSession) -> BiosampleIndex:
    """Mock biosample index dataset."""
    bi_schema = BiosampleIndex.get_schema()

    # Makes arrays of varying length with random integers between 1 and 100
    array_expression = "transform(sequence(1, 1 + floor(rand() * 9)), x -> cast((rand() * 100) as int))"

    data_spec = (
        dg.DataGenerator(
            spark,
            rows=400,
            partitions=4,
            randomSeedMethod="hash_fieldname",
        )
        .withSchema(bi_schema)
        .withColumnSpec(
            "biosampleId",
            expr="cast(id as string)",
        )
        .withColumnSpec("biosampleName", percentNulls=0.1)
        .withColumnSpec("description", percentNulls=0.1)
        .withColumnSpec("xrefs", expr=array_expression, percentNulls=0.1)
        .withColumnSpec("synonyms", expr=array_expression, percentNulls=0.1)
        .withColumnSpec("parents", expr=array_expression, percentNulls=0.1)
        .withColumnSpec("ancestors", expr=array_expression, percentNulls=0.1)
        .withColumnSpec("descendants", expr=array_expression, percentNulls=0.1)
        .withColumnSpec("children", expr=array_expression, percentNulls=0.1)
    )

    return BiosampleIndex(_df=data_spec.build(), _schema=bi_schema)


@pytest.fixture()
def liftover_chain_37_to_38(spark: SparkSession) -> LiftOverSpark:
    """Sample liftover chain file."""
    return LiftOverSpark("tests/gentropy/data_samples/grch37_to_grch38.over.chain")


@pytest.fixture()
def sample_l2g_gold_standard(spark: SparkSession) -> DataFrame:
    """Sample L2G gold standard curation."""
    return spark.read.json(
        "tests/gentropy/data_samples/l2g_gold_standard_curation_sample.json.gz",
    )


@pytest.fixture()
def sample_otp_interactions(spark: SparkSession) -> DataFrame:
    """Sample OTP gene-gene interactions dataset."""
    return spark.read.parquet(
        "tests/gentropy/data_samples/otp_interactions_sample.parquet",
    )


@pytest.fixture()
def mock_l2g_feature_matrix(spark: SparkSession) -> L2GFeatureMatrix:
    """Mock l2g feature matrix dataset."""
    return L2GFeatureMatrix(
        _df=spark.createDataFrame(
            [
                ("1", "gene1", 100.0, None, True),
                ("2", "gene2", 1000.0, 0.0, False),
            ],
            "studyLocusId STRING, geneId STRING, distanceTssMean FLOAT, distanceSentinelTssMinimum FLOAT, goldStandardSet BOOLEAN",
        ),
        with_gold_standard=True,
    )


@pytest.fixture()
def mock_l2g_gold_standard(spark: SparkSession) -> L2GGoldStandard:
    """Mock l2g gold standard dataset."""
    schema = L2GGoldStandard.get_schema()
    data_spec = (
        dg.DataGenerator(
            spark, rows=400, partitions=4, randomSeedMethod="hash_fieldname"
        )
        .withSchema(schema)
        .withColumnSpec(
            "studyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "variantId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "geneId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "traitFromSourceMappedId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "goldStandardSet",
            values=[
                L2GGoldStandard.GS_NEGATIVE_LABEL,
                L2GGoldStandard.GS_POSITIVE_LABEL,
            ],
        )
    )

    return L2GGoldStandard(_df=data_spec.build(), _schema=schema)


@pytest.fixture()
def mock_l2g_predictions(spark: SparkSession) -> L2GPrediction:
    """Mock l2g predictions dataset."""
    schema = L2GPrediction.get_schema()
    data_spec = (
        dg.DataGenerator(
            spark, rows=400, partitions=4, randomSeedMethod="hash_fieldname"
        )
        .withSchema(schema)
        .withColumnSpec(
            "studyLocusId",
            expr="cast(id as string)",
        )
        .withColumnSpec(
            "geneId",
            expr="cast(id as string)",
        )
    )

    return L2GPrediction(_df=data_spec.build(), _schema=schema)


@pytest.fixture()
def sample_data_for_carma() -> list[np.ndarray]:
    """Sample data for fine-mapping by CARMA."""
    ld = pd.read_csv("tests/gentropy/data_samples/01_test_ld.csv", header=None)
    ld = np.array(ld)
    z = pd.read_csv("tests/gentropy/data_samples/01_test_z.csv")
    z = np.array(z.iloc[:, 1])
    pips = pd.read_csv("tests/gentropy/data_samples/01_test_PIPs.txt")
    pips = np.array(pips.iloc[:, 0])
    return [ld, z, pips]


@pytest.fixture()
def sample_data_for_susie_inf() -> list[np.ndarray]:
    """Sample data for fine-mapping by SuSiE-inf."""
    ld = np.loadtxt("tests/gentropy/data_samples/01_test_ld.csv", delimiter=",")
    z = pd.read_csv("tests/gentropy/data_samples/01_test_z.csv")
    z = np.array(z.iloc[:, 1])
    lbf_moments = np.loadtxt("tests/gentropy/data_samples/01_test_lbf_moments.csv")
    lbf_mle = np.loadtxt("tests/gentropy/data_samples/01_test_lbf_mle.csv")
    return [ld, z, lbf_moments, lbf_mle]
