"""DAG that generates a variant index dataset based on several sources."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from common_airflow import (
    create_batch_job,
    create_cluster,
    create_task_spec,
    delete_cluster,
    install_dependencies,
    read_yaml_config,
    shared_dag_args,
    shared_dag_kwargs,
    submit_step,
)
from google.cloud import batch_v1

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.cloud_batch import (
    CloudBatchSubmitJobOperator,
)
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ID = "open-targets-genetics-dev"
REGION = "europe-west1"
GCS_BUCKET = "genetics_etl_python_playground"
CONFIG_FILE_PATH = Path(__file__).parent / "configs" / "variant_sources.yaml"
GENTROPY_DOCKER_IMAGE = "europe-west1-docker.pkg.dev/open-targets-genetics-dev/gentropy-app/gentropy:il-variant-idx"  # TODO: change to dev
VEP_DOCKER_IMAGE = "europe-west1-docker.pkg.dev/open-targets-genetics-dev/gentropy-app/custom_ensembl_vep:dev"
VEP_CACHE_BUCKET = f"gs://{GCS_BUCKET}/vep/cache"

RELEASE = "XX.XX"  # This needs to be updated to the latest release

VCF_DST_PATH = f"gs://{GCS_BUCKET}/{RELEASE}/variant_vcf"
VCF_MERGED_DST_PATH = f"{VCF_DST_PATH}/merged"
VEP_OUTPUT_BUCKET = f"gs://{GCS_BUCKET}/{RELEASE}/vep_output"
VARIANT_INDEX_BUCKET = f"gs://{GCS_BUCKET}/{RELEASE}/variant_index"
GNOMAD_ANNOTATION_PATH = f"gs://{GCS_BUCKET}/static_assets/gnomad_variants"
# Internal parameters for the docker image:
MOUNT_DIR = "/mnt/disks/share"

CLUSTER_NAME = "otg-variant-index"
AUTOSCALING = "eqtl-preprocess"


@task(task_id="vcf_creation")
def create_vcf(**kwargs: Any) -> None:
    """Task that sends the ConvertToVcfStep job to Google Batch.

    Args:
        **kwargs (Any): Keyword arguments
    """
    sources = read_yaml_config(CONFIG_FILE_PATH)
    task_env = [
        batch_v1.Environment(
            variables={
                "SOURCE_NAME": source["name"],
                "SOURCE_PATH": source["location"],
                "SOURCE_FORMAT": source["format"],
            }
        )
        for source in sources["sources_inclusion_list"]
    ]

    commands = [
        "-c",
        rf"poetry run gentropy step=variant_to_vcf step.source_path=$SOURCE_PATH step.source_format=$SOURCE_FORMAT step.vcf_path={VCF_DST_PATH}/$SOURCE_NAME +step.session.extended_spark_conf={{spark.jars:https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar}}",
    ]
    task = create_task_spec(
        GENTROPY_DOCKER_IMAGE, commands, options="-e HYDRA_FULL_ERROR=1"
    )

    batch_task = CloudBatchSubmitJobOperator(
        task_id="vep_batch_job",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=f"vcf-job-{time.strftime('%Y%m%d-%H%M%S')}",
        job=create_batch_job(
            task,
            "VEPMACHINE",
            task_env,
        ),
        deferrable=False,
    )

    batch_task.execute(context=kwargs)


@task(task_id="merge_vcfs")
def merge_vcfs(chunk_size: int = 2000, **kwargs: Any) -> None:
    """Task that merges the information from all the VCF files into a single one so that we only submit one VEP job.

    Args:
        chunk_size (int): Partition size of the merged file. Defaults to 2000.
        **kwargs (Any): Keyword arguments
    """
    ti = kwargs["ti"]
    input_vcfs = [
        f"gs://{GCS_BUCKET}/{listed_file}"
        for listed_file in ti.xcom_pull(
            task_ids="get_vcf_per_source", key="return_value"
        )
    ]
    merged_df = (
        pd.concat(
            pd.read_csv(
                file,
                sep="\t",
                dtype={
                    "#CHROM": str,
                    "POS": int,
                    "ID": str,
                    "REF": str,
                    "ALT": str,
                    "QUAL": str,
                    "FILTER": str,
                    "INFO": str,
                },
            )
            for file in input_vcfs
        )
        .drop_duplicates(subset=["#CHROM", "POS", "REF", "ALT"])
        .sort_values(by=["#CHROM", "POS"])
        .reset_index(drop=True)
    )
    # Partition the merged file into chunks of 2000 variants to run the VEP jobs in parallel
    chunks = 0
    for i in range(0, len(merged_df), chunk_size):
        merged_df[i : i + chunk_size].to_csv(
            f"{VCF_MERGED_DST_PATH}/chunk_{i + 1}-{i + chunk_size}.vcf",
            index=False,
            header=True,
            sep="\t",
        )
        chunks += 1
    expected_chunks_count = len(merged_df) // chunk_size + 1
    assert (
        chunks == expected_chunks_count
    ), f"Expected {expected_chunks_count} chunks but got {chunks} chunks"


@dataclass
class PathManager:
    """It is quite complicated to keep track of all the input/output buckets, the corresponding mounting points prefixes etc..."""

    VCF_INPUT_BUCKET: str
    VEP_OUTPUT_BUCKET: str
    VEP_CACHE_BUCKET: str
    MOUNT_DIR_ROOT: str

    # Derived parameters to find the list of files to process:
    input_path: str | None = None
    input_bucket: str | None = None

    # Derived parameters to initialise the docker image:
    path_dictionary: dict[str, dict[str, str]] | None = None

    # Derived parameters to point to the right mouting points:
    cache_dir: str | None = None
    input_dir: str | None = None
    output_dir: str | None = None

    def __post_init__(self: PathManager) -> None:
        """Build paths based on the input parameters."""
        self.path_dictionary = {
            "input": {
                "remote_path": self.VCF_INPUT_BUCKET.replace("gs://", ""),
                "mount_point": f"{self.MOUNT_DIR_ROOT}/input",
            },
            "output": {
                "remote_path": self.VEP_OUTPUT_BUCKET.replace("gs://", ""),
                "mount_point": f"{self.MOUNT_DIR_ROOT}/output",
            },
            "cache": {
                "remote_path": self.VEP_CACHE_BUCKET.replace("gs://", ""),
                "mount_point": f"{self.MOUNT_DIR_ROOT}/cache",
            },
        }
        # Parameters for fetching files:
        self.input_path = self.VCF_INPUT_BUCKET.replace("gs://", "") + "/"
        self.input_bucket = self.VCF_INPUT_BUCKET.split("/")[2]

        # Parameters for VEP:
        self.cache_dir = f"{self.MOUNT_DIR_ROOT}/cache"
        self.input_dir = f"{self.MOUNT_DIR_ROOT}/input"
        self.output_dir = f"{self.MOUNT_DIR_ROOT}/output"

    def get_mount_config(self) -> list[dict[str, str]]:
        """Return the mount configuration.

        Returns:
            list[dict[str, str]]: The mount configuration.
        """
        assert self.path_dictionary is not None, "Path dictionary not initialized."
        return list(self.path_dictionary.values())


@task(task_id="vep_annotation")
def vep_annotation(pm: PathManager, **kwargs: Any) -> None:
    """Submit a Batch job to annotate VCFs with a local VEP docker image.

    Args:
        pm (PathManager): The path manager with all the required path related information.
        **kwargs (Any): Keyword arguments.
    """
    # Get the filenames to process:
    ti = kwargs["ti"]
    filenames = [
        os.path.basename(os.path.splitext(path)[0])
        for path in ti.xcom_pull(task_ids="get_vep_todo_list", key="return_value")
    ]
    # Stop process if no files was found:
    assert filenames, "No files found to process."

    # Based on the filenames, build the environment variables for the batch job:
    task_env = [
        batch_v1.Environment(
            variables={
                "INPUT_FILE": f"{filename}.vcf",
                "OUTPUT_FILE": f"{filename}.json",
            }
        )
        for filename in filenames
    ]
    # Build the command to run in the container:
    command = [
        "-c",
        rf"vep --cache --offline --format vcf --force_overwrite \
            --no_stats \
            --dir_cache {pm.cache_dir} \
            --input_file {pm.input_dir}/$INPUT_FILE \
            --output_file {pm.output_dir}/$OUTPUT_FILE --json \
            --dir_plugins {pm.cache_dir}/VEP_plugins \
            --sift b \
            --polyphen b \
            --fasta {pm.cache_dir}/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
            --mane_select \
            --appris \
            --hgvsg \
            --pick_order  mane_select,canonical \
            --per_gene \
            --uniprot \
            --check_existing \
            --exclude_null_alleles \
            --canonical \
            --plugin TSSDistance \
            --distance 500000 \
            --plugin LoF,loftee_path:{pm.cache_dir}/VEP_plugins,gerp_bigwig:{pm.cache_dir}/gerp_conservation_scores.homo_sapiens.GRCh38.bw,human_ancestor_fa:{pm.cache_dir}/human_ancestor.fa.gz,conservation_file:/opt/vep/loftee.sql \
            --plugin AlphaMissense,file={pm.cache_dir}/AlphaMissense_hg38.tsv.gz,transcript_match=1 \
            --plugin CADD,snv={pm.cache_dir}/CADD_GRCh38_whole_genome_SNVs.tsv.gz",
    ]
    task = create_task_spec(VEP_DOCKER_IMAGE, command)
    batch_task = CloudBatchSubmitJobOperator(
        task_id="vep_batch_job",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=f"vep-job-{time.strftime('%Y%m%d-%H%M%S')}",
        job=create_batch_job(task, "VEPMACHINE", task_env, pm.get_mount_config()),
        deferrable=False,
    )
    batch_task.execute(context=kwargs)


with DAG(
    dag_id=Path(__file__).stem,
    description="Open Targets Genetics — create VCF file from datasets that contain variant information",
    default_args=shared_dag_args,
    **shared_dag_kwargs,
) as dag:
    pm = PathManager(
        VCF_MERGED_DST_PATH,
        VEP_OUTPUT_BUCKET,
        VEP_CACHE_BUCKET,
        MOUNT_DIR,
    )
    (
        create_vcf()
        >> GCSListObjectsOperator(
            task_id="get_vcf_per_source",
            bucket=GCS_BUCKET,
            prefix=VCF_DST_PATH.replace(f"gs://{GCS_BUCKET}/", ""),
            trigger_rule=TriggerRule.ALL_SUCCESS,
            match_glob="**.csv",
        )
        >> merge_vcfs()
        >> GCSListObjectsOperator(
            task_id="get_vep_todo_list",
            bucket=GCS_BUCKET,
            prefix=VCF_MERGED_DST_PATH.replace(f"gs://{GCS_BUCKET}/", ""),
            trigger_rule=TriggerRule.ALL_SUCCESS,
            match_glob="**.vcf",
        )
        >> vep_annotation(pm)
        >> create_cluster(
            CLUSTER_NAME,
            autoscaling_policy=AUTOSCALING,
            num_workers=4,
            worker_machine_type="n1-highmem-8",
        )
        >> install_dependencies(CLUSTER_NAME)
        >> submit_step(
            cluster_name=CLUSTER_NAME,
            step_id="ot_variant_index",
            task_id="ot_variant_index",
            other_args=[
                f"step.vep_output_json_path={VEP_OUTPUT_BUCKET}",
                f"step.variant_index_path={VARIANT_INDEX_BUCKET}",
                f"step.gnomad_variant_annotations_path={GNOMAD_ANNOTATION_PATH}",
            ],
        )
        >> delete_cluster(CLUSTER_NAME)
    )
