FROM python:3.10-bullseye

RUN apt-get update \
  && apt-get clean \
  && apt-get install -y openjdk-11-jdk \
  && rm -rf /var/lib/apt/lists/*

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    JAVA_HOME=/usr

RUN pip install poetry>=2.0.0
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry config installer.max-workers 10
RUN poetry install --without dev,docs,tests --no-root --no-interaction --no-ansi -vvv && rm -rf $POETRY_CACHE_DIR

COPY src ./src

RUN poetry install --without dev,docs,tests

ENTRYPOINT ["poetry", "run", "gentropy"]
