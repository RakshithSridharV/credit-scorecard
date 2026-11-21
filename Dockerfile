# syntax=docker/dockerfile:1
# Scoring API image (D-026).  Build: make image   Full stack: make up
ARG PYTHON_VERSION=3.12

# ---- stage 1: install the locked dependencies into a virtualenv ----------------
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Copied before the source: this slow layer is rebuilt only when the lock changes,
# not on every code edit.
COPY requirements/serve.lock /tmp/serve.lock
RUN pip install -r /tmp/serve.lock

# ---- stage 2: the runtime image (no pip cache, no build leftovers) -------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src
# PYTHONPATH instead of `pip install .`: config.py finds configs/ relative to the
# source tree (repo root = /app), exactly as in the editable dev install.
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
# Files stay owned by root: the app user can read the model, never modify it.
COPY alembic.ini ./
COPY configs/ configs/
COPY migrations/ migrations/
COPY models/scorecard.json models/reason_reference.json models/
COPY reports/test_evaluation.lock.json reports/
COPY src/ src/
USER app
EXPOSE 8000
# Liveness only (/health). Readiness (/ready) is the orchestrator's call.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request as u; sys.exit(u.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
# One process per container: scale with replicas, not uvicorn workers.
CMD ["uvicorn", "scorecard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
