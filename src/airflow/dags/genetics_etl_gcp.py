"""Generate jinja2 template for workflow."""
from __future__ import annotations

from pathlib import Path

import pendulum
import yaml
from airflow.decorators import dag, task_group
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)
from airflow.utils.trigger_rule import TriggerRule
from airflow_common import generate_create_cluster_task

DAG_ID = "etl_using_external_flat_file"

DAG_DIR = Path(__file__).parent
CONFIG_DIR = "configs"

SOURCES_FILE_NAME = "dag.yaml"
SOURCE_CONFIG_FILE_PATH = DAG_DIR / CONFIG_DIR / SOURCES_FILE_NAME

# Managed cluster
project_id = "open-targets-genetics-dev"
otg_version = "0.1.4"
initialisation_base_path = (
    f"gs://genetics_etl_python_playground/initialisation/{otg_version}"
)
python_cli = f"{initialisation_base_path}/cli.py"
config_name = "my_config"
config_tar = f"{initialisation_base_path}/config.tar.gz"
package_wheel = f"{initialisation_base_path}/otgenetics-{otg_version}-py3-none-any.whl"
initialisation_executable_file = [f"{initialisation_base_path}/initialise_cluster.sh"]
image_version = "2.1"
num_local_ssds = 1
# job
cluster_config_dir = "/config"

default_args = {
    "owner": "Open Targets Data Team",
    # Tell airflow to start one day ago, so that it runs as soon as you upload it
    "start_date": pendulum.now(tz="Europe/London").subtract(days=1),
    # "start_date": pendulum.datetime(2020, 1, 1, tz="Europe/London"),
    "schedule_interval": "@once",
    "project_id": project_id,
    "catchup": False,
    "retries": 3,
}


def generate_pyspark_job_from_dict(
    step: dict,
    cluster_config_dir: str,
    config_name: str,
    cluster_name: str,
) -> DataprocSubmitJobOperator:
    """Generates a pyspark job from dictionary describing step.

    Args:
        step (dict): Dictionary describing step.
        cluster_config_dir (str): Directory containing cluster config.
        config_name (str): Name of config file.
        cluster_name (str): Name of cluster.

    Returns:
        DataprocSubmitJobOperator: Operator for submitting pyspark job.
    """
    return DataprocSubmitJobOperator(
        task_id=f"job-{step['id']}",
        region="europe-west1",
        project_id=project_id,
        job={
            "job_uuid": f"airflow-{step['id']}",
            "reference": {"project_id": project_id},
            "placement": {"cluster_name": cluster_name},
            "pyspark_job": {
                "main_python_file_uri": f"{initialisation_base_path}/cli.py",
                "args": [
                    f"step={ step['id'] }",
                    f"--config-dir={ cluster_config_dir }",
                    f"--config-name={ config_name }",
                ],
                "properties": {
                    "spark.jars": "/opt/conda/miniconda3/lib/python3.10/site-packages/hail/backend/hail-all-spark.jar",
                    "spark.driver.extraClassPath": "/opt/conda/miniconda3/lib/python3.10/site-packages/hail/backend/hail-all-spark.jar",
                    "spark.executor.extraClassPath": "./hail-all-spark.jar",
                    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                    "spark.kryo.registrator": "is.hail.kryo.HailKryoRegistrator",
                },
            },
        },
    )


@dag(
    dag_id=Path(__file__).stem,
    default_args=default_args,
    description="Open Targets Genetics ETL workflow",
    tags=["genetics_etl", "experimental"],
)
def create_dag() -> None:
    """Submit dataproc workflow."""
    start = EmptyOperator(task_id="start")

    source_config_file_path = Path(SOURCE_CONFIG_FILE_PATH)

    end = EmptyOperator(task_id="end", trigger_rule="all_done")

    if source_config_file_path.exists():
        with open(source_config_file_path, "r") as config_file:
            tasks_groups = {}
            steps = yaml.safe_load(config_file)
            for step in steps:
                print(step["id"])

                @task_group(
                    group_id=step["id"],
                    prefix_group_id=True,
                )
                def tgroup(step: dict) -> None:
                    """Task group for step.

                    Args:
                        step (dict): Dictionary describing step.
                    """
                    cluster_name = (
                        f"workflow-otg-cluster-{step['id'].replace('_', '-')}"
                    )
                    create_cluster = generate_create_cluster_task(cluster_name)
                    install_dependencies = DataprocSubmitJobOperator(
                        task_id=f"install_dependencies_{step['id']}",
                        region="europe-west1",
                        project_id=project_id,
                        job={
                            "job_uuid": "airflow-install-dependencies",
                            "reference": {"project_id": project_id},
                            "placement": {"cluster_name": cluster_name},
                            "pig_job": {
                                "jar_file_uris": [
                                    f"gs://genetics_etl_python_playground/initialisation/{otg_version}/install_dependencies_on_cluster.sh"
                                ],
                                "query_list": {
                                    "queries": [
                                        "sh chmod 750 ${PWD}/install_dependencies_on_cluster.sh",
                                        "sh ${PWD}/install_dependencies_on_cluster.sh",
                                    ]
                                },
                            },
                        },
                    )
                    task = generate_pyspark_job_from_dict(
                        step,
                        cluster_config_dir=cluster_config_dir,
                        config_name=config_name,
                        cluster_name=cluster_name,
                    )
                    delete_cluster = DataprocDeleteClusterOperator(
                        task_id=f"delete_cluster_{step['id']}",
                        project_id=project_id,
                        cluster_name=cluster_name,
                        region="europe-west1",
                        trigger_rule=TriggerRule.ALL_DONE,
                        deferrable=True,
                    )
                    create_cluster >> install_dependencies >> task >> delete_cluster

                thisgroup = tgroup(step)
                tasks_groups[step["id"]] = thisgroup
                if "prerequisites" in step:
                    for prerequisite in step["prerequisites"]:
                        print(f"|- {prerequisite}")
                        thisgroup.set_upstream(tasks_groups[prerequisite])

                start >> thisgroup >> end


dag = create_dag()
