# Data

This folder is git-ignored except for this file and the `.gitkeep` placeholders. A pre-commit hook also blocks data files, even if they are force-added.

**Why the data is not in the repository.** *Give Me Some Credit* is distributed by Kaggle under competition rules that do not allow redistribution. Each user downloads it directly.

## Getting the data

1. Sign in to Kaggle and accept the rules of the *Give Me Some Credit* competition.
2. Download `cs-training.csv`.
3. Place it at `data/raw/cs-training.csv`. Nothing else is needed; `cs-test.csv` has no labels and is not used.

## Folder layout

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | `cs-training.csv` exactly as downloaded. Never modified by any code | You |
| `processed/` | Raw train / valid / test splits (70 / 15 / 15, stratified, seed 42) | `python -m scorecard.data.split` |
| `interim/` | The same splits after the cleaning rules D-003 to D-008 | `python -m scorecard.data.clean` |
| `audit/` | Local SQLite decision log, when the API runs without Postgres | the API |

`python -m scorecard.pipelines.train` creates `processed/` and `interim/` from `raw/` in one step.

## The 11 columns

| Column | Meaning |
|---|---|
| `SeriousDlqin2yrs` | **Target.** 1 = 90+ days past due within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Revolving balances / credit limits |
| `age` | Age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days late in the last 2 years |
| `DebtRatio` | Monthly debt payments / monthly income |
| `MonthlyIncome` | Monthly income (may be missing) |
| `NumberOfOpenCreditLinesAndLoans` | Open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Mortgage and real-estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days late in the last 2 years |
| `NumberOfDependents` | Dependents, excluding the borrower (may be missing) |

The schema is enforced in code by `src/scorecard/data/load.py`. See [reports/data_audit.md](../reports/data_audit.md) for the full audit.
