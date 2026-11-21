# Assumptions and limitations

This register lists every number in the project that was **not learned from the data**, where it came from, and what changes if it is wrong. Values live in [configs/config.yaml](../configs/config.yaml). Decision numbers refer to [decisions.md](decisions.md).

**Labels used throughout**

| Label | Meaning |
|---|---|
| **Benchmark** | Taken from a published standard or a widely used industry rule of thumb (source named) |
| **Assumption** | A judgement made for this demo. A lender would replace it with its own figure |
| **Illustrative** | A number chosen or derived only to show the method. It is not a forecast |
| **Data fact** | Read directly from the dataset definition |

> **Scope.** This is a demonstration of the scorecard method on public US consumer-credit data (Kaggle, *Give Me Some Credit*). It is not a model for Indian lending, or for any real lending decision. Money figures are in USD and are Illustrative.

## 1. Business and policy inputs

These drive the approve/decline cutoff (D-018). The cutoff rule is *approve if PD <= break-even PD*, where break-even PD = profit per good / (profit per good + loss per bad).

| Input | Value | Label | Rationale | Effect if wrong |
|---|---|---|---|---|
| Exposure per approved account (EAD) | $5,000 | Illustrative | A round figure for an unsecured consumer line | Scales all profit numbers. It cancels out of the break-even PD, so it does not move the cutoff |
| Yearly margin after funding and servicing | 5% | Assumption | Typical net spread on unsecured credit before credit losses | A higher margin raises the break-even PD, which gives a lower cutoff and more approvals |
| Horizon | 2 years | Data fact | The target is "90+ days past due within 2 years" | Not a choice |
| Loss given default (LGD) | 75% | Assumption | Unsecured consumer credit recovers little after default | A higher LGD gives a higher cutoff (see sensitivity below) |
| Profit per good loan | $500 | Illustrative (derived) | $5,000 x 5% x 2 years | |
| Loss per bad loan | $3,750 | Illustrative (derived) | $5,000 x 75% | |
| Break-even PD | 11.76% | Illustrative (derived) | 500 / (500 + 3,750) | Gives cutoff **549** |

**Sensitivity (valid set)**

| Loss per bad | Break-even PD | Cutoff | Approval rate | Bad rate among approved |
|---|---|---|---|---|
| Halved ($1,875) | 21.05% | 532 | 92.2% | 3.83% |
| Base ($3,750) | 11.76% | 549 | 86.9% | 3.04% |
| Doubled ($7,500) | 6.25% | 569 | 72.6% | 1.88% |

The cutoff moves by about 20 points for a 2x change in loss per bad. Any real use must start by replacing the money inputs with the lender's own.

## 2. Score scale and calibration

| Item | Value | Label | Source and rationale |
|---|---|---|---|
| Base score | 600 | Assumption (industry convention) | Siddiqi, *Credit Risk Scorecards*. A scale convention, not learned |
| Base odds | 50:1 good:bad at 600 | Assumption (industry convention) | Same source |
| Points to double the odds (PDO) | 20 | Assumption (industry convention) | Same source |
| PD floor | 0.05% | Benchmark | Basel III minimum PD for retail exposures |
| Master scale | isotonic, out-of-fold on train | Method choice (D-016) | Fixes curvature in the raw logistic PD while keeping the points additive |

## 3. Feature engineering and selection thresholds

| Item | Value | Label | Rationale |
|---|---|---|---|
| Minimum bin share | 2% of train rows | Assumption | About 2,100 rows per bin, enough for a stable bad rate |
| Maximum numeric bins per feature | 6 | Assumption | Keeps the scorecard readable. Missing gets its own bin |
| WoE smoothing | +0.5 per count | Assumption | Prevents ln(0) on a bin with no bads |
| Minimum IV to keep a feature | 0.02 | Benchmark | Siddiqi rule of thumb: below 0.02 means "not predictive" |
| Maximum pairwise WoE correlation | 0.7 | Benchmark (common practice) | Of a correlated pair, keep the higher-IV feature |
| Maximum VIF | 5 | Benchmark (common practice) | Multicollinearity check after the pairwise one |
| Minimum AUC loss to keep a feature | 0.0005 (5-fold CV on train) | Assumption | A feature that adds less is dropped for simplicity |

## 4. Reason codes (adverse action)

| Item | Value | Label | Rationale |
|---|---|---|---|
| Maximum reasons per decline | 4 | Benchmark | Regulation B commentary: more than four is unlikely to help the applicant |
| Minimum points lost to count as a reason | 5 | Assumption | A quarter of the PDO, about 19% higher odds of bad |
| Reference applicant | train-average points per feature | Method choice (D-019) | "This pulled the applicant below a typical applicant" |

## 5. Monitoring thresholds

| Item | Value | Label |
|---|---|---|
| PSI / CSI stable | < 0.10 | Benchmark (industry rule of thumb, Siddiqi) |
| PSI / CSI investigate | 0.10 - 0.25 | Benchmark (same source) |
| PSI / CSI act | > 0.25 | Benchmark (same source) |
| Approval-rate drift: investigate | > 3 percentage points from train | Assumption |
| Approval-rate drift: act | > 6 percentage points | Assumption |

## 6. Data-cleaning assumptions

Each rule was checked against the data audit ([reports/data_audit.md](../reports/data_audit.md)). It is applied by one function, `clean()`, in training and in the API alike.

| Rule | Assumption behind it | Decision |
|---|---|---|
| `MonthlyIncome` of 0 or 1 is treated as missing | These values are placeholders, not real incomes | D-003 |
| `DebtRatio` becomes missing when income is missing, 0 or 1 | Without income, the field holds a debt amount, not a ratio | D-004 |
| `NumberOfDependents` stays missing | Missing is informative. It gets its own WoE bin instead of being imputed | D-005 |
| Past-due codes 96 / 98 become missing | They are status codes, not counts | D-006 |
| Utilisation above 10 becomes missing | Values far above 1 are mostly dollar balances typed into a ratio field. 1-10 is kept as genuine over-limit borrowing | D-007 |
| Ages under 18 are dropped (one row, age 0); ages above 100 are kept | A credit contract needs an adult; very old ages are rare but possible | D-008 |
| Duplicate rows are kept | No customer ID exists, so identical rows may be different people | D-009 |
| No capping of long tails | WoE binning is already robust to outliers | D-010 |
| No resampling for the 6.7% bad rate | Resampling distorts PDs. The master scale needs true base rates | D-011 |

## 7. Limitations

- **Approved-population bias (no reject inference).** The data contains only borrowers who were granted credit. The model has never seen applicants a lender declined, so its PDs for the riskiest applicants rely on extrapolation. A real build would use reject inference or a period of approving below the cutoff to collect outcomes.
- **Simplified economics.** The profit model ignores:
  - the time value of money;
  - prepayment;
  - partial balances and credit-line utilisation;
  - cross-sell;
  - collection costs beyond LGD.

  The ranking of cutoffs is meaningful; the dollar amounts are Illustrative.
- **One time period.** Train, valid and test are random splits of one snapshot. That is why their PSI is near zero (test score PSI 0.0009). Out-of-time performance is unknown.
- **US data and US regulation.** The reason codes follow ECOA / Regulation B. Other jurisdictions have different adverse-action and fairness rules.
- **Age is a model feature.** Regulation B allows age in an empirically derived, demonstrably sound system, provided applicants 62+ are not disadvantaged. They receive at least as many age points as the best under-62 group (check passed, D-019). A lender may still choose to remove age as a policy decision.
- **The challenger is stronger but not deployed.** LightGBM ranks better (test AUC 0.870 vs 0.860; the gap is significant). The scorecard stays in production for transparency, stable reason codes and ease of validation (D-017). The challenger is the benchmark for shadow monitoring.

## 8. Open risks

| ID | Risk | Status and mitigation |
|---|---|---|
| R-1 | Codes 96 / 98 may be recorded *after* default, which would leak the target | Cannot be verified in this data. Test performance reported with and without the 40 affected rows (AUC 0.8602 vs 0.8588): immaterial |
| R-2 | API behaviour for 96 / 98 | **Resolved (D-023).** Accepted and mapped to missing, exactly as in training. Covered by tests |
| R-3 | Income withholding: not reporting income earns about +3 points in one case | Small effect. A lender should require or verify income (policy, not model) |
| R-4 | Config drift: the cutoff is recomputed from `config.yaml`, which is not hash-checked at start-up | The config is reviewed in git, and every decision returns the cutoff used. Full fix: pin the expected cutoff in the lock |
| R-5 | Retention: Regulation B requires records for 25 months | No purge job yet |
| R-6 | No idempotency key: a client retry can create two records | Add a client `application_id` with a unique constraint |
| R-7 | No authentication or TLS on the API | Bound to localhost. In production it sits behind an API gateway (mTLS / OAuth) |
| R-8 | Artifact hashes include Windows line endings | Stored byte-for-byte via `.gitattributes`. Writing LF or canonical JSON would need a re-lock (a governance event) |
| R-9 | The skew test needs the data, so it cannot run in CI | Run `make check` locally before any release |
| R-10 | The image includes training-only libraries (906 MB) | Split `train` and `serve` dependency groups |
