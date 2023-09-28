PROJECT_ID ?= open-targets-genetics-dev
REGION ?= europe-west1
APP_NAME ?= $$(cat pyproject.toml| grep name | cut -d" " -f3 | sed  's/"//g')
VERSION_NO ?= $$(poetry version --short)
CLEAN_VERSION_NO := $(shell echo "$(VERSION_NO)" | tr -cd '[:alnum:]')
BUCKET_NAME=gs://genetics_etl_python_playground/initialisation/${VERSION_NO}/
BUCKET_COMPOSER_DAGS=gs://europe-west1-ot-workflows-fe147745-bucket/dags/

.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

.DEFAULT_GOAL := help

help: ## This is help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## Clean up prior to building
	@rm -Rf ./dist

setup-dev: SHELL:=/bin/bash
setup-dev: ## Setup development environment
	@. utils/install_dependencies.sh
	@. python utils/install_gcs_connector.py
check: ## Lint and format code
	@echo "Linting..."
	@poetry run ruff src/otg .
	@echo "Formatting..."
	@poetry run black src/otg .
	@poetry run isort src/otg .

test: ## Run tests
	@echo "Running Tests..."
	@poetry run pytest --doctest-modules --cov=src/ --cov-report=xml

build-documentation: ## Create local server with documentation
	@echo "Building Documentation..."
	@poetry run mkdocs serve

create-dev-cluster: ## Spin up a simple dataproc cluster with all dependencies for development purposes
	@${MAKE} build
	@echo "Creating Dataproc Cluster"
	@gcloud config set project ${PROJECT_ID}
	@gcloud dataproc clusters create "ot-genetics-dev-${CLEAN_VERSION_NO}" \
		--image-version 2.1 \
		--region ${REGION} \
		--master-machine-type n1-standard-16 \
		--initialization-actions=gs://genetics_etl_python_playground/initialisation/${VERSION_NO}/initialise_cluster.sh \
		--metadata="PACKAGE=gs://genetics_etl_python_playground/initialisation/${VERSION_NO}/otgenetics-${VERSION_NO}-py3-none-any.whl,CONFIGTAR=gs://genetics_etl_python_playground/initialisation/${VERSION_NO}/config.tar.gz" \
		--single-node \
		--enable-component-gateway

build: clean ## Build Python package with dependencies
	@gcloud config set project ${PROJECT_ID}
	@echo "Packaging Code and Dependencies for ${APP_NAME}-${VERSION_NO}"
	@rm -rf ./dist
	@poetry build
	@tar -czf dist/config.tar.gz config/
	@echo "Uploading to Dataproc"
	@gsutil cp src/otg/cli.py ${BUCKET_NAME}
	@gsutil cp ./dist/${APP_NAME}-${VERSION_NO}-py3-none-any.whl ${BUCKET_NAME}
	@gsutil cp ./dist/config.tar.gz ${BUCKET_NAME}
	@gsutil cp ./utils/initialise_cluster.sh ${BUCKET_NAME}
	@gsutil -m cp -r 'dags/*' ${BUCKET_COMPOSER_DAGS}
