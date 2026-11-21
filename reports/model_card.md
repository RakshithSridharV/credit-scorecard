# Model card: Give Me Some Credit scorecard

Follows the structure of *Model Cards for Model Reporting* (Mitchell et al., 2019), extended with the items a credit model validator expects. Decision numbers (D-0xx) refer to [docs/decisions.md](../docs/decisions.md). Money figures are **Illustrative** (see [docs/assumptions.md](../docs/assumptions.md)).

## 1. Model details

| | |
|---|---|
| Name | Give Me Some Credit application scorecard |
| Version | `27f4de0069b8` (first 12 hex characters of the SHA-256 of `models/scorecard.json`) |
| Tested | 2026-11-20, one-time test evaluation, fingerprinted in `reports/test_evaluation.lock.json` |
| Type | Points-based scorecard: monotonic WoE binning + unpenalised logistic regression, with an isotonic score-to-PD master scale |
| Scale | 600 points = 50:1 good:bad odds; 20 points double the odds (industry convention) |
| Output | Integer score, calibrated PD, approve/decline at cutoff 549, up to 4 reason codes on declines |
| Benchmark | LightGBM challenger (`models/challenger_lgbm.txt`). Not used for decisions |
| Code | `src/scorecard/models/scorecard.py`; served by `src/scorecard/api/` |
| Owner | Repository author |

## 2. Intended use

- **Primary use.** Demonstrating an end-to-end, audit-ready application scorecard. It ranks US consumer applicants for **unsecured revolving credit** by their probability of becoming 90+ days past due within 2 years.
- **Intended users.** Reviewers of credit-risk modelling practice: model developers, validators, and engineers studying the build and serving pattern.
- **Decision supported.** Approve or decline at a break-even PD cutoff, with adverse-action reasons for every decline.

## 3. Out-of-scope uses

- **Any real credit decision.** The money inputs are Illustrative, and the model has not been validated on a lender's own applicants.
- **Lending in India or any non-US market.** The data comes from US borrowers and the reason codes follow US rules (ECOA / Regulation B). Bureau fields, regulation and borrower behaviour all differ elsewhere.
- **Other products.** Secured loans, mortgages, business or SME credit, and collections scoring.
- **Credit limits, pricing or provisioning** without separate calibration and validation for that purpose.
- **Populations unlike the training data.** Applicants a lender would previously have declined (see section 9, approved-population bias).

## 4. Training data

| | |
|---|---|
| Source | Kaggle *Give Me Some Credit* (`cs-training.csv`), US consumer credit. Not redistributed (licence) |
| Size | 150,000 borrowers; 1 row dropped (age 0) |
| Target | `SeriousDlqin2yrs = 1`: 90+ days past due within 2 years. Bad rate 6.68% |
| Split | 70 / 15 / 15 train / valid / test, stratified on the target, seed 42 (D-012) |
| Use of each split | **Train**: everything fitted. **Valid**: every choice. **Test**: scored once (D-020) |
| Cleaning | One function used in training and serving alike. Placeholders and special codes become missing; nothing is imputed (D-003 to D-008) |
| Known data issues | No customer ID, 609 exact duplicates kept (D-009); special codes 96 / 98 of unknown meaning (risk R-1) |

## 5. Features

7 of the 10 bureau fields are used, chosen by IV, redundancy checks and cross-validated backward elimination on train (D-015).

| Feature | IV (train) | Direction |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 1.133 | Higher utilisation, fewer points |
| NumberOfTimes90DaysLate | 0.842 | Any 90+ day lateness costs about 41 points |
| NumberOfTime30-59DaysPastDueNotWorse | 0.736 | More late payments, fewer points |
| NumberOfTime60-89DaysPastDueNotWorse | 0.574 | Any, fewer points |
| age | 0.247 | Older, more points (see section 8) |
| DebtRatio | 0.080 | Higher ratio, fewer points |
| NumberRealEstateLoansOrLines | 0.044 | Non-linear, binned |

Not used: `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans` and `NumberOfDependents`. They did not improve cross-validated AUC enough. `MonthlyIncome` is still read at scoring time, because cleaning uses it to decide whether `DebtRatio` is a real ratio (D-004). Missing values get their own bin and points.

## 6. Performance

**Headline (test, 22,500 rows, never used for any choice)**

| Metric | Valid | Test | Test 95% CI |
|---|---|---|---|
| AUC | 0.8592 | 0.8602 | 0.8507 - 0.8702 |
| Gini | 0.7184 | 0.7203 | 0.7014 - 0.7405 |
| KS | 0.5616 | 0.5689 | 0.5483 - 0.5910 |
| Brier score | 0.0501 | 0.0497 | |
| Mean PD / bad rate | 0.990 | 0.983 | |

**Calibration.** Across test score deciles 4-10 (the riskier 70%, where almost all defaults are), predicted PD / observed bad rate is between 0.93 and 1.11. In the three safest deciles, observed bad rates are only 0.5-0.8%. There, a handful of defaults moves the ratio, and it ranges from 0.93 to 1.44. Full table in `reports/test_evaluation.md`.

**Decision at cutoff 549 (test).**
- Approves 87.0% of applicants.
- Bad rate among approved: 2.97% (vs 6.68% overall).
- Declines 61.4% of future defaulters and 9.5% of good applicants.
- Value over approve-all: **+$1.10M per 10,000 applicants (Illustrative)**.

**Benchmark.** LightGBM reaches test AUC 0.8702. Its gap over the scorecard is +0.010 (95% CI 0.007 - 0.013). The scorecard is kept for exact reason codes and easier validation (D-017).

**Robustness.** Without the 40 test rows with codes 96 / 98, AUC is 0.8588 vs 0.8602: the possible leakage (R-1) does not drive performance.

## 7. Performance by subgroup (valid)

From `reports/subgroup_performance.md`, produced by `python -m scorecard.evaluation.subgroups`. It uses valid because test is not reopened for new analyses.

| Group | Share | Bad rate | Mean PD | AUC | Approval | Approval ratio | Goods declined |
|---|---|---|---|---|---|---|---|
| Age 18-29 | 6.2% | 11.16% | 12.18% | 0.795 | 76.2% | **0.79** | 18.2% |
| Age 30-44 | 25.4% | 9.74% | 9.42% | 0.839 | 79.8% | 0.83 | 15.3% |
| Age 45-61 | 40.4% | 6.73% | 6.77% | 0.854 | 86.6% | 0.90 | 10.1% |
| Age 62+ | 28.0% | 2.86% | 2.61% | 0.857 | 96.0% | 1.00 | 2.9% |
| Income reported | 78.8% | 7.00% | 7.08% | 0.851 | 85.6% | 0.94 | 10.8% |
| Income not reported | 21.3% | 5.50% | 4.88% | 0.888 | 91.4% | 1.00 | 5.9% |

**Findings**

1. **Approval rates for 18-29 year-olds are just under the four-fifths screening threshold** (0.79 against 0.8, a **Benchmark** rule of thumb). The gap mostly follows observed risk: the group's bad rate is 3.9x that of applicants 62+. The model also over-predicts this group's PD slightly (12.18% vs 11.16%), but that difference is within sampling noise (about 1.2 standard errors on 1,389 rows). A lender would review this before use. One option is to measure the approval impact of removing `age` from the model.
2. **The model ranks young applicants less well** (AUC 0.795 vs 0.85+ for older groups). Their files are thinner, so the bureau fields carry less signal. Declines for this group are less certain.
3. **Applicants who do not report income are slightly under-predicted** (PD 4.88% vs bad rate 5.50%, about 1.9 standard errors, borderline). Together with the small points gain from a missing `DebtRatio` (D-023), this supports a policy of requiring or verifying income (risk R-3).
4. **Applicants 62+ are not disadvantaged.** They have the highest approval rate, and they receive at least as many age points as the best under-62 group (Regulation B check, D-019).

## 8. Explainability and fairness

- **Reason codes.** Every decline lists up to 4 reasons, ranked by points lost against the train-average applicant. A loss must be at least 5 points to count (D-019). On test declines, the first reason is high utilisation in 52% of cases and 90+ days late in 30%.
- **Age.** ECOA lists age as a prohibited basis. Regulation B permits it in an empirically derived, demonstrably and statistically sound scoring system, provided applicants 62+ are not scored down for age. The check passes (finding 4). The 18-29 approval ratio (finding 1) is disclosed for review. Age appears as a reason on 21% of test declines (in the top 4).
- **Protected attributes not in the data.** Race, sex, marital status and national origin are absent, so they cannot be used directly. They also cannot be tested for proxy effects. A lender would run a fair-lending analysis with proxy methods (e.g. BISG) before use.

## 9. Limitations

- **Approved-population bias.** There is no reject inference, so performance on applicants a lender previously declined is unknown.
- **One time period.** All splits come from one snapshot. Out-of-time and through-the-cycle stability are untested.
- **Illustrative economics.** The cutoff depends on the assumed margin, exposure and LGD. A 2x change in loss per bad moves it about 20 points.
- **Hand-written reports.** The cutoff, reason codes and subgroup tables are regenerated by the pipeline. This card and the docs are written by hand and must be updated after any rebuild.

## 10. Monitoring and maintenance

- **Baseline.** Train score bands and feature bins are frozen in `models/monitoring_baseline.json`.
- **Signals.**
  - Score PSI and feature CSI: < 0.10 stable, 0.10 - 0.25 investigate, > 0.25 act (**Benchmark**).
  - Approval-rate drift: > 3 points investigate, > 6 points act (**Assumption**).

  The overall status is the worst of the three. Test PSI is 0.0009 (stable).
- **Shadow benchmark.** The LightGBM challenger is scored alongside the scorecard. A widening gap is a trigger for model review.
- **Audit log.** Every decision is stored with its inputs and `model_version`, so it can be re-scored for disputes and used for outcome analysis.
- **Change control.**
  - `python -m scorecard.pipelines.train` reproduces the tested model byte for byte (D-022).
  - The API refuses any model file that differs from the tested one (D-023).
  - A new model needs a new one-time test evaluation and a new version.

## 11. References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT* '19.
- Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed. Wiley. Source of the scaling, IV and PSI conventions.
- Board of Governors of the Federal Reserve System (2011). *SR 11-7: Guidance on Model Risk Management*.
- 12 CFR Part 1002 (Regulation B, Equal Credit Opportunity Act), including 1002.6(b)(2) on age and 1002.9 on adverse-action notices.
- Basel Committee on Banking Supervision. *Basel III* framework: 0.05% PD floor for retail exposures.
