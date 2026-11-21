# Credit scorecard: from raw bureau data to an audited scoring API

An application credit scorecard built the way a bank would need it:
- monotonic WoE binning and logistic regression, calibrated to a PD;
- an approve/decline cutoff derived from break-even economics;
- Regulation B reason codes on every decline;
- served by a FastAPI service that refuses untested models and records every decision in an append-only audit log.

The data is Kaggle's *Give Me Some Credit* (150,000 US borrowers). **This is a method demonstration on US data, not a model for Indian lending or for any real credit decision.**

![CI](https://github.com/RakshithSridharV/credit-scorecard/actions/workflows/ci.yml/badge.svg)

## Results (held-out test set, scored once)

| | |
|---|---|
| AUC / Gini / KS | **0.860** / 0.720 / 0.569 (AUC 95% CI 0.851 - 0.870) |
| Calibration (mean PD / bad rate) | 0.983 |
| Cutoff 549 | approves 87.0%; bad rate among approved 2.97% vs 6.68% overall |
| Value vs approving everyone | **+$1.10M per 10,000 applicants (Illustrative)** |
| LightGBM challenger | AUC 0.870. Kept as a benchmark; the scorecard ships for exact reason codes |
| Score stability (PSI, test vs train) | 0.0009 (stable) |

Money figures are **Illustrative**: they depend on assumed exposure, margin and loss inputs ([docs/assumptions.md](docs/assumptions.md)). Every number in the project is labelled Benchmark, Assumption, Illustrative or Data fact.

## What is in it

- **No leakage.** The data is split before anything is fitted. Everything is learned on train and chosen on valid, and test is scored exactly once. Its fingerprints are locked in [reports/test_evaluation.lock.json](reports/test_evaluation.lock.json).
- **Reproducible.** One command rebuilds every artifact from the raw CSV and proves it is byte-identical to the tested model.
- **One decision path.** The API uses the same `clean()` as training. A skew test shows identical scores offline and online.
- **Fail-safe serving.** Strict input contract (422 errors that never echo applicant data), fail-fast start-up, and fail-closed audit writes: no record means no decision.
- **Least-privilege deployment.** Docker image runs as a non-root user. A migrate-once job manages the schema. The API's Postgres role can only `SELECT` and `INSERT`.
- **Governed.** 32 recorded decisions with alternatives, a [model card](reports/model_card.md) including subgroup performance, monitoring with PSI / CSI, and CI on every push.

## Quick start

Requires Python 3.12. Docker is optional.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[api,dev,app]"
python -m pre_commit install         # hooks on commit, tests on push
```

**Data.** Download `cs-training.csv` from the Kaggle competition *Give Me Some Credit* and place it in `data/raw/`. The licence forbids redistribution, so the data is never committed (see [data/README.md](data/README.md)).

**Rebuild and verify the model**

```bash
python -m scorecard.pipelines.train      # all stages; compares hashes with the test lock
python -m pytest -q                      # unit, API and integration tests
```

**Run the API locally (SQLite audit log)**

```bash
python -m alembic upgrade head
python -m uvicorn scorecard.api.main:app --port 8000
# open http://127.0.0.1:8000/docs
```

**Underwriter what-if app (simulation only, records nothing)**

```bash
python -m streamlit run app/streamlit_app.py     # http://127.0.0.1:8501
```

Enter an application to see the score, PD, outcome, decline reasons and the points behind each feature. Then change fields to see how the score would move. For example, paying utilisation down from 0.95 to 0.25 adds 32 points to the README decline below, which is still short of the cutoff.

**Run the full stack (Postgres) with Docker**

```bash
cp .env.example .env                     # then set real passwords
make up                                  # db -> migrate -> api, waits until healthy
make smoke                               # end-to-end checks + DB privilege check
make down
```

## Example

```bash
curl -s -X POST http://127.0.0.1:8000/score -H "content-type: application/json" -d '{
  "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 27,
  "NumberOfTime30-59DaysPastDueNotWorse": 2, "DebtRatio": 0.3, "MonthlyIncome": 5400,
  "NumberOfOpenCreditLinesAndLoans": 8, "NumberOfTimes90DaysLate": 1,
  "NumberRealEstateLoansOrLines": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2}'
```

```json
{
  "decision_id": "3f6c…",
  "score": 487,
  "pd": 0.460431654676259,
  "decision": "decline",
  "cutoff": 549,
  "reasons": [
    {"code": "DLQ_90", "text": "Serious delinquency: payment 90 or more days past due", "points_lost": 38.7},
    {"code": "UTIL_HIGH", "text": "Balances on revolving accounts are high relative to credit limits", "points_lost": 28.1},
    {"code": "DLQ_30", "text": "Payments 30-59 days past due in the last two years", "points_lost": 26.1},
    {"code": "AGE", "text": "Age of applicant", "points_lost": 8.9}
  ],
  "model_version": "27f4de0069b8"
}
```

The same applicant with utilisation 0.25, age 45 and no late payments scores **594** and is approved (PD 2.0%), with no reasons attached. Reasons are only required for adverse action.

## Documentation

| Document | Contents |
|---|---|
| [docs/report/credit_scorecard_report.pdf](docs/report/credit_scorecard_report.pdf) | **Start here.** 16-page model development report: the whole project in one document |
| [docs/architecture.md](docs/architecture.md) | Pipeline, scoring service, audit table, deployment, quality gates |
| [docs/business_case.md](docs/business_case.md) | Policy outcome, Illustrative value, sensitivity, scorecard vs challenger, KPIs |
| [docs/assumptions.md](docs/assumptions.md) | Every non-learned number with its source label, limitations, open risks |
| [docs/decisions.md](docs/decisions.md) | Decision log D-001 to D-032: what, why, alternatives |
| [reports/model_card.md](reports/model_card.md) | Intended use, performance, subgroups, fairness, monitoring |
| [reports/age_ablation.md](reports/age_ablation.md) | Experiment: the scorecard without `age`, and who bears the cost |
| [reports/](reports/) | Generated reports for each pipeline stage |

## Project layout

```
app/          Streamlit what-if app for underwriters (simulation only)
configs/      config.yaml: seeds, thresholds, business inputs
src/scorecard/
  data/       load, audit, clean, split
  features/   binning, WoE, IV, selection
  models/     scorecard, LightGBM challenger
  evaluation/ metrics, calibration, cutoff, subgroups, age experiment, PSI/CSI
  explain/    reason codes
  pipelines/  train (rebuild + verify), evaluate (one-time test run)
  api/        scoring core, FastAPI app, audit-log database
migrations/   Alembic schema migrations
models/       fitted artifacts (JSON), fingerprinted
reports/      generated reports, test lock, model card
tests/        unit/, api/, integration/
```

## Tech stack

Python 3.12, pandas, scikit-learn, LightGBM, FastAPI, Pydantic, Streamlit, SQLAlchemy, Alembic, Postgres, Docker, pytest, ruff, pre-commit, GitHub Actions.

## Note on development environment

The development laptop runs Windows with Smart App Control. Some dependencies are pinned to releases it allows, and pre-commit hooks run as `python -m` modules rather than downloaded executables. Neither affects Linux or CI.
