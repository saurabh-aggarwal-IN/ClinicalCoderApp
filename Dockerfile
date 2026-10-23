# One image, three harness commits. The build args pin the identity that the
# harness reports back to the orchestrator.
FROM nousresearch/hermes-agent:latest

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN python -m venv /opt/harness-venv \
    && /opt/harness-venv/bin/python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN /opt/harness-venv/bin/python -m pip install -r /tmp/requirements.txt

COPY . /app
RUN chown -R 10000:10000 /app \
    && chmod 0755 /app/docker/entrypoint.sh

ARG GIT_COMMIT=unknown
ARG HARNESS_ID=h1
ARG HARNESS_VERSION=1.0.0
ENV GIT_COMMIT=${GIT_COMMIT} \
    HARNESS_ID=${HARNESS_ID} \
    HARNESS_VERSION=${HARNESS_VERSION} \
    HARNESS_IMPLEMENTATION=python \
    HERMES_BIN=hermes \
    HERMES_HOME_ROOT=/tmp/clinical-coder-hermes \
    API_HOST=0.0.0.0 \
    API_PORT=8080

USER 10000:10000
WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["/opt/harness-venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).read()"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD []
