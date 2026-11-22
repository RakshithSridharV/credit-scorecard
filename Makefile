# One entry point per task, so nobody has to remember module names (D-022).
# Windows: run from Git Bash, or install make (e.g. `winget install GnuWin32.Make`).
PY ?= python

.PHONY: help hooks install train rebuild evaluate rehearse dupcheck migrate serve app lock image up down smoke test lint format check

help:
	@echo "install   editable install with dev tools"
	@echo "hooks     install the git hooks (pre-commit on commit, tests on push)"
	@echo "train     rebuild every artifact from data/raw, verify against the test lock"
	@echo "rebuild   rebuild without verifying (deliberate redevelopment)"
	@echo "rehearse  run the test evaluation on VALID (writes nothing)"
	@echo "evaluate  ONE-TIME test evaluation (refuses a second run)"
	@echo "dupcheck  D-009 duplicates check on valid + test (recorded in the lock, D-033)"
	@echo "migrate   bring the audit database to the latest schema (alembic upgrade head)"
	@echo "serve     run the scoring API on http://127.0.0.1:8000 (docs at /docs)"
	@echo "app       underwriter what-if app on http://127.0.0.1:8501 (simulation only)"
	@echo "lock      re-resolve the Docker dependency lock (requirements/serve.lock)"
	@echo "image     build the API image"
	@echo "up        start Postgres + migrate + API in Docker (needs .env)"
	@echo "smoke     check the running API end to end, and that the DB refuses UPDATE"
	@echo "down      stop the stack (add V=1 to also delete the database volume)"
	@echo "test      unit tests"
	@echo "lint      ruff lint + format check"
	@echo "check     lint + test (what CI runs)"

install:
	$(PY) -m pip install -e ".[api,dev,app]"

hooks:
	$(PY) -m pre_commit install

train:
	$(PY) -m scorecard.pipelines.train

rebuild:
	$(PY) -m scorecard.pipelines.train --no-verify

rehearse:
	$(PY) -m scorecard.pipelines.evaluate --rehearse

evaluate:
	$(PY) -m scorecard.pipelines.evaluate

dupcheck:
	$(PY) -m scorecard.evaluation.duplicates

migrate:
	$(PY) -m alembic upgrade head

serve: migrate
	$(PY) -m uvicorn scorecard.api.main:app --host 127.0.0.1 --port 8000

app:
	$(PY) -m streamlit run app/streamlit_app.py --server.address 127.0.0.1

# ---- Docker (D-026) ------------------------------------------------------------
# The lock is resolved INSIDE the same Linux image the API runs on, so the pins
# match the platform that installs them (not this Windows laptop).
# Built from a copy: installing from the mounted folder leaves build/ and egg-info behind.
LOCK_CMD = cp -r /src /tmp/build \
  && pip install -q --disable-pip-version-check '/tmp/build[api,postgres]' \
  && (echo '\# Resolved by make lock inside python:3.12-slim (D-026). Do not edit by hand.'; \
      pip freeze --exclude-editable | grep -v '^scorecard') > requirements/serve.lock

lock:
	docker run --rm -v "$(CURDIR):/src" -w /src python:3.12-slim sh -c "$(LOCK_CMD)"

image:
	docker compose build

up:
	docker compose up -d --build --wait api

smoke:
	$(PY) scripts/smoke.py
	@echo "--- the API's database role must not be able to change history:"
	@docker compose exec -T db psql -U scorecard_app -d scorecard -c "UPDATE decisions SET score = 999" 2>&1 	  | grep -q "permission denied" && echo "PASS  UPDATE refused for scorecard_app" 	  || (echo "FAIL  scorecard_app could UPDATE decisions" && exit 1)

down:
	docker compose down $(if $(V),-v,)

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests migrations scripts app
	ruff format --check src tests migrations scripts app

format:
	ruff check --fix src tests migrations scripts app
	ruff format src tests migrations scripts app

check: lint test  # same as CI's first job, minus the pre-commit file-hygiene hooks
