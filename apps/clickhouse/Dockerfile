# Application « ClickHouse » : mêmes fonctions que « base », avec ClickHouse pour les passages (BM25 en SQL),
# les documents triés par date et les traces. Construction, depuis la RACINE du dépôt :
#   docker build -f apps/clickhouse/Dockerfile -t objectagentrag-clickhouse .
# Les versions des dépendances ne sont pas figées : après un premier build validé, figez-les
# (pip freeze > requirements.lock) pour des builds reproductibles.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AGENTIC_DATA_DIR=/data AGENTIC_CONFIG=/app/config.toml AGENTIC_SERVER_HOST=0.0.0.0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentic_search ./agentic_search
COPY apps/clickhouse/config.toml ./config.toml
RUN pip install ".[all]" \
    && useradd --create-home --uid 10001 agent \
    && mkdir /data /corpus && chown agent /data

USER agent
VOLUME ["/data", "/corpus"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=3)"
ENTRYPOINT ["agentic-search"]
CMD ["serve"]
