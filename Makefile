SHELL := /bin/bash
PROJECT_ID ?= open-targets-genetics-dev
REGION ?= europe-west1
APP_NAME ?= $$(cat pyproject.toml | grep -m 1 "name" | cut -d" " -f3 | sed  's/"//g')
PACKAGE_VERSION ?= $(shell grep -m 1 'version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
USER_SAFE ?= $(shell echo $(USER) | tr '[:upper:]' '[:lower:]')
CLUSTER_TIMEOUT ?= 60m
# NOTE: git rev-parse will always return the HEAD if it sits in the tag,
# this way we can distinguish the tag vs branch name
ifeq ($(shell git rev-parse --abbrev-ref HEAD),HEAD)
	REF ?= $(shell git describe --exact-match --tags)
else
	REF ?= $(shell git rev-parse --abbrev-ref HEAD)
endif

CLEAN_PACKAGE_VERSION := $(shell echo "$(PACKAGE_VERSION)" | tr -cd '[:alnum:]')
BUCKET_NAME=gs://genetics_etl_python_playground/initialisation

.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

.DEFAULT_GOAL := help

help: ## This is help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## Clean up prior to building
	@rm -Rf ./dist

setup-dev: SHELL := $(shell echo $${SHELL})
setup-dev:  ## Setup development environment
	@. utils/install_dependencies.sh
	@echo "Run . ${HOME}/.$(notdir $(SHELL))rc to finish setup"

check: ## Lint and format code
	@echo "Linting API..."
	@uv run ruff check src/gentropy .
	@echo "Linting docstrings..."
	@uv run pydoclint --config=pyproject.toml src
	@uv run pydoclint --config=pyproject.toml --skip-checking-short-docstrings=true tests

test: ## Run tests
	@echo "Running Tests..."
	@uv run pytest

build-documentation: ## Create local server with documentation
	@echo "Building Documentation..."
	@uv run mkdocs serve

sync-cluster-init-script: ## Synchronize the cluster inicialisation actions script to google cloud
	@echo "Syncing install_dependencies_on_cluster.sh to ${BUCKET_NAME}"
	@gcloud storage cp utils/install_dependencies_on_cluster.sh ${BUCKET_NAME}/install_dependencies_on_cluster.sh

sync-gentropy-cli-script: ## Synchronize the gentropy cli script
	@echo "Syncing gentropy cli script to ${BUCKET_NAME}"
	@gcloud storage cp src/gentropy/cli.py ${BUCKET_NAME}/cli.py

create-dev-cluster: sync-cluster-init-script sync-gentropy-cli-script ## Spin up a simple dataproc cluster with all dependencies for development purposes
	@echo "Making sure the cluster can reference to ${REF} branch to install gentropy..."
	@./utils/clean_status.sh ${REF} || (echo "ERROR: Commit and push local changes, to have up to date cluster"; exit 1)
	@echo "Creating Dataproc Dev Cluster"
	gcloud config set project ${PROJECT_ID}
	gcloud dataproc clusters create "ot-genetics-dev-${CLEAN_PACKAGE_VERSION}-$(USER_SAFE)" \
		--image-version 2.2 \
		--region ${REGION} \
		--master-machine-type n1-standard-2 \
		--metadata="GENTROPY_REF=${REF}" \
		--initialization-actions=${BUCKET_NAME}/install_dependencies_on_cluster.sh \
		--secondary-worker-type spot \
		--worker-machine-type n1-standard-4 \
		--public-ip-address \
		--worker-boot-disk-size 500 \
		--autoscaling-policy="projects/${PROJECT_ID}/regions/${REGION}/autoscalingPolicies/otg-etl" \
		--optional-components=JUPYTER \
		--enable-component-gateway \
		--labels team=open-targets,subteam=gentropy,created_by=${USER_SAFE},environment=development, \
		--max-idle=${CLUSTER_TIMEOUT}

update-dev-cluster: build ## Reinstalls the package on the dev-cluster
	@echo "Updating Dataproc Dev Cluster"
	@gcloud config set project ${PROJECT_ID}
	gcloud dataproc jobs submit pig --cluster="ot-genetics-dev-${CLEAN_PACKAGE_VERSION}" \
		--region ${REGION} \
		--jars=${BUCKET_NAME}/install_dependencies_on_cluster.sh \
		-e='sh chmod 750 $${PWD}/install_dependencies_on_cluster.sh; sh $${PWD}/install_dependencies_on_cluster.sh'

build: clean ## Build Python package with dependencies
	@uv build
