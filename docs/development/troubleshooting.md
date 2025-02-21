---
title: Troubleshooting
---

# Troubleshooting

## BLAS/LAPACK

If you see errors related to BLAS/LAPACK libraries, see [this StackOverflow post](https://stackoverflow.com/questions/69954587/no-blas-lapack-libraries-found-when-installing-scipy) for guidance.

## UV

The default python version and gentropy dependencies are managed by [uv](https://docs.astral.sh/uv/). To perform a fresh installation run `make setup-dev`.

## Adding new dependencies or updating existing ones

To add new dependencies or update existing ones, you need to update the `pyproject.toml` file. This can be done automatically with `uv add ${package}` command. Refer to the [uv documentation](https://docs.astral.sh/uv/) for more information.

## Java

Officially, PySpark requires Java version 8, or 11, 17. To support hail (gentropy dependency) it is recommended to use Java 11.

## Pre-commit

If you see an error message thrown by pre-commit, which looks like this (`SyntaxError: Unexpected token '?'`), followed by a JavaScript traceback, the issue is likely with your system NodeJS version.

One solution which can help in this case is to upgrade your system NodeJS version. However, this may not always be possible. For example, Ubuntu repository is several major versions behind the latest version as of July 2023.

Another solution which helps is to remove Node, NodeJS, and npm from your system entirely. In this case, pre-commit will not try to rely on a system version of NodeJS and will install its own, suitable one.

On Ubuntu, this can be done using `sudo apt remove node nodejs npm`, followed by `sudo apt autoremove`. But in some cases, depending on your existing installation, you may need to also manually remove some files. See [this StackOverflow answer](https://stackoverflow.com/a/41057802) for guidance.

## MacOS

Some functions on MacOS may throw a java error:

`python3.10/site-packages/py4j/protocol.py:326: Py4JJavaError`

This can be resolved by adding the follow line to your `~/.zshrc`:

`export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

## Creating development dataproc cluster (OT users only)

!!! info "Requirements"

    To create the cluster, you need to auth to the google cloud

    ```bash
    gcloud auth login
    ```

To start dataproc cluster in the development mode run.

```bash
make create-dev-cluster REF=dev
```

`REF` - remote branch available at the [gentropy repository](https://github.com/opentargets/gentropy)

During cluster [initialization actions](https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/init-actions#important_considerations_and_guidelines) the `utils/install_dependencies_on_cluster.sh` script is run, that installs `gentropy` package from the remote repository by using VCS support, hence it does not require the **gentropy package whl artifact** to be prepared in the Google Cloud Storage before the make command can be run.

Check details how to make a package installable by VCS in [pip documentation](https://pip.pypa.io/en/stable/topics/vcs-support/).

!!! note "How `create-dev-cluster` works"

    This command will work, provided you have done one of:

    - run `make create-dev-cluster REF=dev`, since the REF is requested, the cluster will attempt to install it from the remote repository.
    - run `make create-dev-cluster` without specifying the REF or specifying REF that points to your local branch will request branch name you are checkout on your local repository, if any changes are pending locally, the cluster can not be created, it requires stashing or pushing the changes to the remote.

    The command will create a new dataproc cluster with the following configuration:

    - package installed from the requested **REF** (for example `dev` or `feature/xxx`)
    - uv installed in the cluster (to speed up the installation and dependency resolution process)
    - cli script to run gentropy steps

!!! tip "Dataproc cluster timeout"

    By default the cluster will **delete itself** when running for **60 minutes after the last submitted job to the cluster was successfully completed** (running jobs interactively via Jupyter or Jupyter lab is not treated as submitted job). To preserve the cluster for arbitrary period (**for instance when the cluster is used only for interactive jobs**) increase the cluster timeout:

    ```bash
    make create-dev-cluster CLUSTER_TIMEOUT=1d REF=dev # 60m 1h 1d (by default 60m)
    ```

    For the reference on timeout format check [gcloud documentation](https://cloud.google.com/sdk/gcloud/reference/dataproc/clusters/create#--max-idle)
