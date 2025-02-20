"""Interface for application configuration."""

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypedDict

from hail import __file__ as hail_location
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class SessionConfig:
    """Session configuration."""

    start_hail: bool = False
    write_mode: str = "errorifexists"
    spark_uri: str = "local[*]"
    hail_home: str = os.path.dirname(hail_location)
    extended_spark_conf: dict[str, str] | None = field(default_factory=dict[str, str])
    output_partitions: int = 200
    _target_: str = "gentropy.common.session.Session"


@dataclass
class StepConfig:
    """Base step configuration."""

    session: SessionConfig
    defaults: list[Any] = field(
        default_factory=lambda: [{"session": "base_session"}, "_self_"]
    )


@dataclass
class ColocalisationConfig(StepConfig):
    """Colocalisation step configuration."""

    credible_set_path: str = MISSING
    coloc_path: str = MISSING
    colocalisation_method: str = MISSING
    colocalisation_method_params: dict[str, Any] = field(default_factory=dict[str, Any])
    _target_: str = "gentropy.colocalisation.ColocalisationStep"


@dataclass
class BiosampleIndexConfig(StepConfig):
    """Biosample index step configuration."""

    cell_ontology_input_path: str = MISSING
    uberon_input_path: str = MISSING
    efo_input_path: str = MISSING
    biosample_index_path: str = MISSING
    _target_: str = "gentropy.biosample_index.BiosampleIndexStep"


@dataclass
class GWASCatalogStudyCurationConfig(StepConfig):
    """GWAS Catalog study curation step configuration."""

    catalog_study_files: list[str] = MISSING
    catalog_ancestry_files: list[str] = MISSING
    gwas_catalog_study_curation_out: str = MISSING
    gwas_catalog_study_curation_file: str = MISSING
    _target_: str = "gentropy.gwas_catalog_study_curation.GWASCatalogStudyCurationStep"


@dataclass
class GWASCatalogStudyIndexGenerationStep(StepConfig):
    """GWAS Catalog study index generation."""

    catalog_study_files: list[str] = MISSING
    catalog_ancestry_files: list[str] = MISSING
    study_index_path: str = MISSING
    gwas_catalog_study_curation_file: str | None = None
    sumstats_qc_path: str | None = None
    _target_: str = (
        "gentropy.gwas_catalog_study_index.GWASCatalogStudyIndexGenerationStep"
    )


@dataclass
class GWASCatalogTopHitIngestionConfig(StepConfig):
    """GWAS Catalog ingestion step configuration."""

    catalog_study_files: list[str] = MISSING
    catalog_ancestry_files: list[str] = MISSING
    catalog_associations_file: str = MISSING
    variant_annotation_path: str = MISSING
    catalog_studies_out: str = MISSING
    catalog_associations_out: str = MISSING
    _target_: str = "gentropy.gwas_catalog_top_hits.GWASCatalogTopHitIngestionStep"


@dataclass
class GWASCatalogSumstatsPreprocessConfig(StepConfig):
    """GWAS Catalog sumstat preprocess step configuration."""

    raw_sumstats_path: str = MISSING
    out_sumstats_path: str = MISSING
    _target_: str = (
        "gentropy.gwas_catalog_sumstat_preprocess.GWASCatalogSumstatsPreprocessStep"
    )


@dataclass
class FoldXVariantAnnotationConfig(StepConfig):
    """Step to ingest FoldX amino acid variation data."""

    foldx_dataset_path: str = MISSING
    plddt_threshold: float = 0.7
    annotation_path: str = MISSING

    _target_: str = "gentropy.foldx_ingestion.FoldXIngestionStep"


@dataclass
class EqtlCatalogueConfig(StepConfig):
    """eQTL Catalogue step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    eqtl_catalogue_paths_imported: str = MISSING
    eqtl_catalogue_study_index_out: str = MISSING
    eqtl_catalogue_credible_sets_out: str = MISSING
    mqtl_quantification_methods_blacklist: list[str] = field(default_factory=lambda: [])
    eqtl_lead_pvalue_threshold: float = 1e-3
    _target_: str = "gentropy.eqtl_catalogue.EqtlCatalogueStep"


@dataclass
class FinngenStudiesConfig(StepConfig):
    """FinnGen study index step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    finngen_study_index_out: str = MISSING
    finngen_phenotype_table_url: str = "https://r11.finngen.fi/api/phenos"
    finngen_release_prefix: str = "FINNGEN_R11_"
    finngen_summary_stats_url_prefix: str = (
        "gs://finngen-public-data-r11/summary_stats/finngen_R11_"
    )
    finngen_summary_stats_url_suffix: str = ".gz"
    efo_curation_mapping_url: str = "https://raw.githubusercontent.com/opentargets/curation/24.09.1/mappings/disease/manual_string.tsv"
    # https://www.finngen.fi/en/access_results#:~:text=Total%20sample%20size%3A%C2%A0453%2C733%C2%A0(254%2C618%C2%A0females%20and%C2%A0199%2C115%20males)
    sample_size: int = 453733
    _target_: str = "gentropy.finngen_studies.FinnGenStudiesStep"


@dataclass
class FinngenFinemappingConfig(StepConfig):
    """FinnGen fine mapping ingestion step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    finngen_susie_finemapping_snp_files: str = (
        "gs://finngen-public-data-r11/finemap/full/susie/*.snp.bgz"
    )
    finngen_susie_finemapping_cs_summary_files: str = (
        "gs://finngen-public-data-r11/finemap/summary/*SUSIE.cred.summary.tsv"
    )
    finngen_finemapping_out: str = MISSING
    finngen_finemapping_lead_pvalue_threshold: float = 1e-5
    finngen_release_prefix: str = "FINNGEN_R11"
    _target_: str = (
        "gentropy.finngen_finemapping_ingestion.FinnGenFinemappingIngestionStep"
    )


@dataclass
class LDIndexConfig(StepConfig):
    """LD index step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    ld_index_out: str = MISSING
    min_r2: float = 0.5
    ld_matrix_template: str = "gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.{POP}.common.adj.ld.bm"
    ld_index_raw_template: str = "gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.{POP}.common.ld.variant_indices.ht"
    liftover_ht_path: str = "gs://gcp-public-data--gnomad/release/2.1.1/liftover_grch38/ht/genomes/gnomad.genomes.r2.1.1.sites.liftover_grch38.ht"
    grch37_to_grch38_chain_path: str = (
        "gs://hail-common/references/grch37_to_grch38.over.chain.gz"
    )
    ld_populations: list[str] = field(
        default_factory=lambda: [
            "afr",  # African-American
            "amr",  # American Admixed/Latino
            "eas",  # East Asian
            "fin",  # Finnish
            "nfe",  # Non-Finnish European
        ]
    )
    _target_: str = "gentropy.gnomad_ingestion.LDIndexStep"


@dataclass
class LDBasedClumpingConfig(StepConfig):
    """LD based clumping step configuration."""

    study_locus_input_path: str = MISSING
    study_index_path: str = MISSING
    ld_index_path: str = MISSING
    clumped_study_locus_output_path: str = MISSING
    _target_: str = "gentropy.ld_based_clumping.LDBasedClumpingStep"


@dataclass
class IntervalConfig(StepConfig):
    """Interval step configuration."""

    target_index_path: str = MISSING
    liftover_chain_file_path: str = MISSING
    interval_sources: dict[str, str] = MISSING
    processed_interval_path: str = MISSING
    _target_: str = "gentropy.intervals.IntervalStep"


@dataclass
class LocusToGeneConfig(StepConfig):
    """Locus to gene step configuration."""

    run_mode: str = MISSING
    credible_set_path: str = MISSING
    feature_matrix_path: str = MISSING
    predictions_path: str | None = None
    l2g_threshold: float | None = 0.05
    variant_index_path: str | None = None
    model_path: str | None = None
    gold_standard_curation_path: str | None = None
    gene_interactions_path: str | None = None
    interval_path: str | None = None
    features_list: list[str] = field(
        default_factory=lambda: [
            # max CLPP for each (study, locus, gene) aggregating over a specific qtl type
            "eQtlColocClppMaximum",
            "pQtlColocClppMaximum",
            "sQtlColocClppMaximum",
            # max H4 for each (study, locus, gene) aggregating over a specific qtl type
            "eQtlColocH4Maximum",
            "pQtlColocH4Maximum",
            "sQtlColocH4Maximum",
            # max CLPP for each (study, locus, gene) aggregating over a specific qtl type and in relation with the mean in the vicinity
            "eQtlColocClppMaximumNeighbourhood",
            "pQtlColocClppMaximumNeighbourhood",
            "sQtlColocClppMaximumNeighbourhood",
            # max H4 for each (study, locus, gene) aggregating over a specific qtl type and in relation with the mean in the vicinity
            "eQtlColocH4MaximumNeighbourhood",
            "pQtlColocH4MaximumNeighbourhood",
            "sQtlColocH4MaximumNeighbourhood",
            # distance to gene footprint
            "distanceSentinelFootprint",
            "distanceSentinelFootprintNeighbourhood",
            "distanceFootprintMean",
            "distanceFootprintMeanNeighbourhood",
            # distance to gene tss
            "distanceTssMean",
            "distanceTssMeanNeighbourhood",
            "distanceSentinelTss",
            "distanceSentinelTssNeighbourhood",
            # vep
            "vepMaximum",
            "vepMaximumNeighbourhood",
            "vepMean",
            "vepMeanNeighbourhood",
            # intervals
            "pchicMean",
            "pchicMeanNeighbourhood",
            "enhTssCorrelationMean",
            "enhTssCorrelationMeanNeighbourhood",
            "dhsPmtrCorrelationMean",
            "dhsPmtrCorrelationMeanNeighbourhood",
            # other
            "geneCount500kb",
            "proteinGeneCount500kb",
            "credibleSetConfidence",
        ]
    )
    hyperparameters: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "ccp_alpha": 0,
            "learning_rate": 0.1,
            "min_samples_leaf": 5,
            "min_samples_split": 5,
            "subsample": 1,
        }
    )
    wandb_run_name: str | None = None
    hf_hub_repo_id: str | None = "opentargets/locus_to_gene"
    hf_model_commit_message: str | None = "chore: update model"
    download_from_hub: bool = True
    cross_validate: bool = False
    _target_: str = "gentropy.l2g.LocusToGeneStep"


@dataclass
class LocusToGeneFeatureMatrixConfig(StepConfig):
    """Locus to gene feature matrix step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "extended_spark_conf": {
                "spark.driver.memory": "48g",
                "spark.executor.memory": "48g",
                "spark.sql.shuffle.partitions": "800",
            }
        }
    )
    credible_set_path: str = MISSING
    variant_index_path: str | None = None
    colocalisation_path: str | None = None
    study_index_path: str | None = None
    target_index_path: str | None = None
    interval_path: str | None = None
    feature_matrix_path: str = MISSING
    features_list: list[str] = field(
        default_factory=lambda: [
            # max CLPP for each (study, locus, gene) aggregating over a specific qtl type
            "eQtlColocClppMaximum",
            "pQtlColocClppMaximum",
            "sQtlColocClppMaximum",
            # max H4 for each (study, locus, gene) aggregating over a specific qtl type
            "eQtlColocH4Maximum",
            "pQtlColocH4Maximum",
            "sQtlColocH4Maximum",
            # max CLPP for each (study, locus, gene) aggregating over a specific qtl type and in relation with the mean in the vicinity
            "eQtlColocClppMaximumNeighbourhood",
            "pQtlColocClppMaximumNeighbourhood",
            "sQtlColocClppMaximumNeighbourhood",
            # max H4 for each (study, locus, gene) aggregating over a specific qtl type and in relation with the mean in the vicinity
            "eQtlColocH4MaximumNeighbourhood",
            "pQtlColocH4MaximumNeighbourhood",
            "sQtlColocH4MaximumNeighbourhood",
            # distance to gene footprint
            "distanceSentinelFootprint",
            "distanceSentinelFootprintNeighbourhood",
            "distanceFootprintMean",
            "distanceFootprintMeanNeighbourhood",
            # distance to gene tss
            "distanceTssMean",
            "distanceTssMeanNeighbourhood",
            "distanceSentinelTss",
            "distanceSentinelTssNeighbourhood",
            # vep
            "vepMaximum",
            "vepMaximumNeighbourhood",
            "vepMean",
            "vepMeanNeighbourhood",
            # intervals
            "pchicMean",
            "pchicMeanNeighbourhood",
            "enhTssCorrelationMean",
            "enhTssCorrelationMeanNeighbourhood",
            "dhsPmtrCorrelationMean",
            "dhsPmtrCorrelationMeanNeighbourhood",
            # other
            "geneCount500kb",
            "proteinGeneCount500kb",
            "credibleSetConfidence",
            "isProteinCoding",
        ]
    )
    _target_: str = "gentropy.l2g.LocusToGeneFeatureMatrixStep"


@dataclass
class PICSConfig(StepConfig):
    """PICS step configuration."""

    study_locus_ld_annotated_in: str = MISSING
    picsed_study_locus_out: str = MISSING
    _target_: str = "gentropy.pics.PICSStep"


@dataclass
class UkbPppEurConfig(StepConfig):
    """UKB PPP (EUR) ingestion step configuration."""

    raw_study_index_path_from_tsv: str = MISSING
    raw_summary_stats_path: str = MISSING
    tmp_variant_annotation_path: str = MISSING
    variant_annotation_path: str = MISSING
    study_index_output_path: str = MISSING
    summary_stats_output_path: str = MISSING
    _target_: str = "gentropy.ukb_ppp_eur_sumstat_preprocess.UkbPppEurStep"


@dataclass
class FinngenUkbMetaConfig(StepConfig):
    """FinnGen UKB meta-analysis ingestion step configuration."""

    raw_study_index_path_from_tsv: str = MISSING
    raw_summary_stats_path: str = MISSING
    tmp_variant_annotation_path: str = MISSING
    variant_annotation_path: str = MISSING
    study_index_output_path: str = MISSING
    summary_stats_output_path: str = MISSING
    _target_: str = "gentropy.finngen_ukb_meta.FinngenUkbMetaIngestionStep"


@dataclass
class GnomadVariantConfig(StepConfig):
    """Gnomad variant ingestion step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    variant_annotation_path: str = MISSING
    gnomad_genomes_path: str = (
        "gs://gcp-public-data--gnomad/release/4.1/ht/joint/gnomad.joint.v4.1.sites.ht/"
    )
    gnomad_variant_populations: list[str] = field(
        default_factory=lambda: [
            "afr",  # African-American
            "amr",  # American Admixed/Latino
            "ami",  # Amish ancestry
            "asj",  # Ashkenazi Jewish
            "eas",  # East Asian
            "fin",  # Finnish
            "nfe",  # Non-Finnish European
            "mid",  # Middle Eastern
            "sas",  # South Asian
            "remaining",  # Other
        ]
    )
    _target_: str = "gentropy.gnomad_ingestion.GnomadVariantIndexStep"


@dataclass
class PanUKBBConfig(StepConfig):
    """Pan UKB variant ingestion step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    pan_ukbb_ht_path: str = "gs://panukbb-ld-matrixes/ukb-diverse-pops-public-build-38/UKBB.{POP}.ldadj.variant.b38"
    pan_ukbb_bm_path: str = "gs://panukbb-ld-matrixes/UKBB.{POP}.ldadj"
    ukbb_annotation_path: str = "gs://panukbb-ld-matrixes/UKBB.{POP}.aligned.parquet"
    pan_ukbb_pops: list[str] = field(
        default_factory=lambda: [
            "AFR",  # African
            "CSA",  # Central/South Asian
            "EUR",  # European
        ]
    )
    _target_: str = "gentropy.pan_ukb_ingestion.PanUKBBVariantIndexStep"


@dataclass
class LOFIngestionConfig(StepConfig):
    """Step configuration for the ingestion of Loss-of-Function variant data generated by OTAR2075."""

    lof_curation_dataset_path: str = MISSING
    lof_curation_variant_annotations_path: str = MISSING

    _target_: str = "gentropy.lof_curation_ingestion.LOFIngestionStep"


@dataclass
class VariantIndexConfig(StepConfig):
    """Variant index step configuration."""

    class _ConsequenceToPathogenicityScoreMap(TypedDict):
        """Typing definition for CONSEQUENCE_TO_PATHOGENICITY_SCORE."""

        id: str
        label: str
        score: float

    session: Any = field(
        default_factory=lambda: {
            "start_hail": False,
        }
    )
    vep_output_json_path: str = MISSING
    variant_index_path: str = MISSING
    variant_annotations_path: list[str] | None = None
    hash_threshold: int = 300
    consequence_to_pathogenicity_score: ClassVar[
        list[_ConsequenceToPathogenicityScoreMap]
    ] = [
        {"id": "SO_0001575", "label": "splice_donor_variant", "score": 1.0},
        {"id": "SO_0001589", "label": "frameshift_variant", "score": 1.0},
        {"id": "SO_0001574", "label": "splice_acceptor_variant", "score": 1.0},
        {"id": "SO_0001587", "label": "stop_gained", "score": 1.0},
        {"id": "SO_0002012", "label": "start_lost", "score": 1.0},
        {"id": "SO_0001578", "label": "stop_lost", "score": 1.0},
        {"id": "SO_0001893", "label": "transcript_ablation", "score": 1.0},
        {"id": "SO_0001822", "label": "inframe_deletion", "score": 0.66},
        {
            "id": "SO_0001818",
            "label": "protein_altering_variant",
            "score": 0.66,
        },
        {"id": "SO_0001821", "label": "inframe_insertion", "score": 0.66},
        {
            "id": "SO_0001787",
            "label": "splice_donor_5th_base_variant",
            "score": 0.66,
        },
        {"id": "SO_0001583", "label": "missense_variant", "score": 0.66},
        {"id": "SO_0001567", "label": "stop_retained_variant", "score": 0.33},
        {"id": "SO_0001630", "label": "splice_region_variant", "score": 0.33},
        {"id": "SO_0002019", "label": "start_retained_variant", "score": 0.33},
        {
            "id": "SO_0002169",
            "label": "splice_polypyrimidine_tract_variant",
            "score": 0.33,
        },
        {
            "id": "SO_0001626",
            "label": "incomplete_terminal_codon_variant",
            "score": 0.33,
        },
        {"id": "SO_0001819", "label": "synonymous_variant", "score": 0.33},
        {
            "id": "SO_0002170",
            "label": "splice_donor_region_variant",
            "score": 0.33,
        },
        {"id": "SO_0001624", "label": "3_prime_UTR_variant", "score": 0.1},
        {"id": "SO_0001623", "label": "5_prime_UTR_variant", "score": 0.1},
        {"id": "SO_0001627", "label": "intron_variant", "score": 0.1},
        {
            "id": "SO_0001619",
            "label": "non_coding_transcript_variant",
            "score": 0.0,
        },
        {"id": "SO_0001580", "label": "coding_sequence_variant", "score": 0.0},
        {"id": "SO_0001632", "label": "downstream_gene_variant", "score": 0.0},
        {"id": "SO_0001631", "label": "upstream_gene_variant", "score": 0.0},
        {
            "id": "SO_0001792",
            "label": "non_coding_transcript_exon_variant",
            "score": 0.0,
        },
        {"id": "SO_0001620", "label": "mature_miRNA_variant", "score": 0.0},
        {"id": "SO_0001060", "label": "intergenic_variant", "score": 0.0},
    ]
    amino_acid_change_annotations: list[str] = MISSING

    _target_: str = "gentropy.variant_index.VariantIndexStep"


@dataclass
class ConvertToVcfStepConfig(StepConfig):
    """Variant to VCF step configuration."""

    source_paths: list[str] = MISSING
    source_formats: list[str] = MISSING
    output_path: str = MISSING
    partition_size: int = 2000
    _target_: str = "gentropy.variant_index.ConvertToVcfStep"


@dataclass
class LocusBreakerClumpingConfig(StepConfig):
    """Locus breaker clumping step configuration."""

    summary_statistics_input_path: str = MISSING
    clumped_study_locus_output_path: str = MISSING
    lbc_baseline_pvalue: float = 1e-5
    lbc_distance_cutoff: int = 250_000
    lbc_pvalue_threshold: float = 1e-8
    lbc_flanking_distance: int = 100_000
    large_loci_size: int = 1_500_000
    wbc_clump_distance: int = 500_000
    wbc_pvalue_threshold: float = MISSING
    collect_locus: bool = False
    remove_mhc: bool = True
    _target_: str = "gentropy.locus_breaker_clumping.LocusBreakerClumpingStep"


@dataclass
class WindowBasedClumpingStepConfig(StepConfig):
    """Window-based clumping step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    summary_statistics_input_path: str = MISSING
    study_locus_output_path: str = MISSING
    gwas_significance: float = 1e-8
    distance: int = 500_000
    collect_locus: bool = False
    collect_locus_distance: int = 500_000
    inclusion_list_path: str | None = None
    _target_: str = "gentropy.window_based_clumping.WindowBasedClumpingStep"


@dataclass
class FinemapperConfig(StepConfig):
    """SuSiE fine-mapper step configuration."""

    session: Any = field(
        default_factory=lambda: {
            "start_hail": True,
        }
    )
    study_index_path: str = MISSING
    study_locus_manifest_path: str = MISSING
    study_locus_index: int = MISSING
    max_causal_snps: int = MISSING
    lead_pval_threshold: float = MISSING
    purity_mean_r2_threshold: float = MISSING
    purity_min_r2_threshold: float = MISSING
    cs_lbf_thr: float = MISSING
    sum_pips: float = MISSING
    susie_est_tausq: bool = MISSING
    run_carma: bool = MISSING
    carma_tau: float = MISSING
    run_sumstat_imputation: bool = MISSING
    carma_time_limit: int = MISSING
    imputed_r2_threshold: float = MISSING
    ld_score_threshold: float = MISSING
    ld_min_r2: float = MISSING
    _target_: str = "gentropy.susie_finemapper.SusieFineMapperStep"


@dataclass
class SummaryStatisticsQCStepConfig(StepConfig):
    """GWAS QC step configuration."""

    gwas_path: str = MISSING
    output_path: str = MISSING
    pval_threshold: float = MISSING
    _target_: str = "gentropy.sumstat_qc_step.SummaryStatisticsQCStep"


@dataclass
class CredibleSetQCStepConfig(StepConfig):
    """Credible set quality control step configuration."""

    credible_sets_path: str = MISSING
    output_path: str = MISSING
    p_value_threshold: float = 1e-5
    purity_min_r2: float = 0.01
    clump: bool = False
    ld_index_path: str | None = None
    study_index_path: str | None = None
    ld_min_r2: float | None = 0.8
    n_partitions: int | None = 200
    _target_: str = "gentropy.credible_set_qc.CredibleSetQCStep"


@dataclass
class StudyValidationStepConfig(StepConfig):
    """Configuration of the study index validation step.

    The study indices are read from multiple location, therefore we are expecting a list of paths.
    """

    study_index_path: list[str] = MISSING
    target_index_path: str = MISSING
    disease_index_path: str = MISSING
    biosample_index_path: str = MISSING
    valid_study_index_path: str = MISSING
    invalid_study_index_path: str = MISSING
    invalid_qc_reasons: list[str] = MISSING
    _target_: str = "gentropy.study_validation.StudyValidationStep"


@dataclass
class LocusToGeneEvidenceStepConfig(StepConfig):
    """Configuration of the locus to gene evidence step."""

    locus_to_gene_predictions_path: str = MISSING
    credible_set_path: str = MISSING
    study_index_path: str = MISSING
    evidence_output_path: str = MISSING
    locus_to_gene_threshold: float = 0.05
    _target_: str = "gentropy.l2g.LocusToGeneEvidenceStep"


@dataclass
class LocusToGeneAssociationsStepConfig(StepConfig):
    """Configuration of the locus to gene association step."""

    evidence_input_path: str = MISSING
    disease_index_path: str = MISSING
    direct_associations_output_path: str = MISSING
    indirect_associations_output_path: str = MISSING
    _target_: str = "gentropy.l2g.LocusToGeneAssociationsStep"


@dataclass
class StudyLocusValidationStepConfig(StepConfig):
    """Configuration of the study index validation step.

    The study locus datasets are read from multiple location, therefore we are expecting a list of paths.
    """

    study_index_path: str = MISSING
    study_locus_path: list[str] = MISSING
    target_index_path: str = MISSING
    valid_study_locus_path: str = MISSING
    invalid_study_locus_path: str = MISSING
    invalid_qc_reasons: list[str] = MISSING
    trans_qtl_threshold: int = MISSING
    _target_: str = "gentropy.study_locus_validation.StudyLocusValidationStep"


@dataclass
class Config:
    """Application configuration."""

    # this is unfortunately verbose due to @dataclass limitations
    defaults: list[Any] = field(default_factory=lambda: ["_self_", {"step": MISSING}])
    step: StepConfig = MISSING
    datasets: dict[str, str] = field(default_factory=dict)


def register_config() -> None:
    """Register configuration."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(group="step/session", name="base_session", node=SessionConfig)
    cs.store(group="step", name="colocalisation", node=ColocalisationConfig)
    cs.store(group="step", name="eqtl_catalogue", node=EqtlCatalogueConfig)
    cs.store(group="step", name="biosample_index", node=BiosampleIndexConfig)
    cs.store(
        group="step",
        name="gwas_catalog_study_curation",
        node=GWASCatalogStudyCurationConfig,
    )
    cs.store(
        group="step",
        name="gwas_catalog_study_index",
        node=GWASCatalogStudyIndexGenerationStep,
    )
    cs.store(
        group="step",
        name="gwas_catalog_sumstat_preprocess",
        node=GWASCatalogSumstatsPreprocessConfig,
    )
    cs.store(
        group="step",
        name="gwas_catalog_top_hit_ingestion",
        node=GWASCatalogTopHitIngestionConfig,
    )
    cs.store(group="step", name="ld_based_clumping", node=LDBasedClumpingConfig)
    cs.store(group="step", name="ld_index", node=LDIndexConfig)
    cs.store(group="step", name="locus_to_gene", node=LocusToGeneConfig)
    cs.store(
        group="step",
        name="locus_to_gene_feature_matrix",
        node=LocusToGeneFeatureMatrixConfig,
    )
    cs.store(group="step", name="finngen_studies", node=FinngenStudiesConfig)

    cs.store(
        group="step",
        name="finngen_finemapping_ingestion",
        node=FinngenFinemappingConfig,
    )

    cs.store(group="step", name="pics", node=PICSConfig)
    cs.store(group="step", name="gnomad_variants", node=GnomadVariantConfig)
    cs.store(group="step", name="ukb_ppp_eur_sumstat_preprocess", node=UkbPppEurConfig)
    cs.store(group="step", name="lof_curation_ingestion", node=LOFIngestionConfig)
    cs.store(group="step", name="variant_index", node=VariantIndexConfig)
    cs.store(group="step", name="variant_to_vcf", node=ConvertToVcfStepConfig)
    cs.store(
        group="step", name="window_based_clumping", node=WindowBasedClumpingStepConfig
    )
    cs.store(group="step", name="susie_finemapping", node=FinemapperConfig)
    cs.store(
        group="step", name="summary_statistics_qc", node=SummaryStatisticsQCStepConfig
    )
    cs.store(
        group="step", name="locus_breaker_clumping", node=LocusBreakerClumpingConfig
    )
    cs.store(
        group="step",
        name="credible_set_validation",
        node=StudyLocusValidationStepConfig,
    )
    cs.store(
        group="step",
        name="study_validation",
        node=StudyValidationStepConfig,
    )
    cs.store(
        group="step",
        name="locus_to_gene_evidence",
        node=LocusToGeneEvidenceStepConfig,
    )
    cs.store(
        group="step",
        name="locus_to_gene_associations",
        node=LocusToGeneAssociationsStepConfig,
    )
    cs.store(group="step", name="finngen_ukb_meta_ingestion", node=FinngenUkbMetaConfig)
    cs.store(group="step", name="credible_set_qc", node=CredibleSetQCStepConfig)
    cs.store(group="step", name="foldx_integration", node=FoldXVariantAnnotationConfig)
