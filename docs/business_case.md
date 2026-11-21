# Business case

**Summary.** On held-out test data, the scorecard at cutoff 549 approves 87% of applicants and cuts the bad rate among approved accounts from 6.68% to 2.97%. It declines 61% of future defaulters and 9.5% of good customers. Under the stated money assumptions, that is worth about **$1.1M per 10,000 applicants (Illustrative)** compared with approving everyone. Every decision is explainable, comes with Regulation B reason codes, and is recorded in an append-only audit log.

> All money figures are **Illustrative**. They come from the demo inputs in [assumptions.md](assumptions.md) (EAD $5,000, margin 5% a year, LGD 75%), not from a lender's books. Rates and model metrics are measured on the dataset. This is a US-data method demo, not an Indian lending model.

## 1. The problem

An unsecured lender has to decide, for each applicant, whether expected interest income outweighs expected credit loss. Three things make manual or ad-hoc rules weak:

- **Losses are asymmetric.** One default costs about 7.5 good loans' worth of profit. With the demo inputs, a bad loan loses $3,750 and a good loan earns $500 (Illustrative).
- **Defaults are rare.** 6.7% of borrowers in the data go 90+ days past due within 2 years (data fact). Without a good ranking, cutting losses means turning away many good customers.
- **Declines must be explained.** Under ECOA / Regulation B, every decline needs specific principal reasons. An unexplainable model cannot be used for adverse action without extra work.

## 2. What the model delivers (test set, never used for any choice)

| Measure | Value | Type |
|---|---|---|
| AUC / Gini / KS | 0.860 / 0.720 / 0.569 | Measured (95% CI for AUC: 0.851 - 0.870) |
| Calibration: mean predicted PD / observed bad rate | 0.983 | Measured. PDs can be used for pricing and provisioning |
| Score stability, test vs train (PSI) | 0.0009 | Measured (stable < 0.10) |
| Valid -> test performance change | AUC 0.859 -> 0.860 | Measured: no overfitting to the choices made on valid |

## 3. Policy outcome at the chosen cutoff (549)

The cutoff is set where the calibrated PD equals the break-even PD of 11.76% (Illustrative, derived from the inputs).

| Per 10,000 applicants (test) | Approve all | Scorecard, cutoff 549 |
|---|---|---|
| Approval rate | 100% | 87.0% |
| Bad rate among approved | 6.68% | 2.97% |
| Share of bads declined | 0% | 61.4% |
| Share of goods declined | 0% | 9.5% |
| Profit (Illustrative) | about $2.16M | about $3.25M |
| **Value of the scorecard (Illustrative)** | | **+$1.10M** |

*From the evaluation report: $3,254,333 - $2,159,111 = +$1,095,222 per 10,000 (Illustrative). On valid, the figure was +$1,061,556 (Illustrative), so the result held up on unseen data.*

**Scaling (Illustrative).** At 100,000 applications a year the same rates give about +$11M a year. This is linear scaling of a demo figure, not a forecast.

**Why 549 and not the most profitable point on valid?** The best cutoff found directly on valid was 552, which earns 0.4% more profit. That gap is sample noise. The break-even rule is a policy a credit committee can reason about and re-derive when costs change. A cutoff tuned to one sample is neither.

## 4. How sensitive the value is to the assumptions

| Scenario (loss per bad) | Cutoff | Approval rate | Profit per 10,000 (Illustrative) |
|---|---|---|---|
| Halved ($1,875) | 532 | 92.2% | $3.77M |
| Base ($3,750) | 549 | 86.9% | $3.22M |
| Doubled ($7,500) | 569 | 72.6% | $2.54M |

(Valid set.) The model is the same in all three rows. Only the business inputs change. Re-pricing risk appetite means changing one line of config, with no retraining.

## 5. Benefits beyond the headline number

- **Consistency and speed.** The same inputs always give the same decision, in about 16 ms per request. Underwriter time goes to referrals, not routine cases.
- **Compliance by design.**
  - Each decline carries up to 4 reason codes taken straight from the points table. On test declines, high utilisation is the first reason in 52% of cases and 90+ days late in 30%.
  - The age treatment passes the Regulation B check for applicants aged 62+.
- **Auditability.** Each decision is stored with:
  - its raw inputs;
  - the model version (a fingerprint of the exact tested file);
  - the cutoff and the reasons.

  Any dispute can be re-scored. The service refuses to start with a model that differs from the one tested.
- **Controlled change.** A one-command rebuild reproduces the tested model byte for byte. Schema changes go through migrations, and CI blocks untested changes.
- **Early warning.** PSI, CSI and approval-rate monitoring against a frozen baseline show when the applicant mix shifts, before losses appear.

## 6. Scorecard vs machine-learning challenger

A tuned LightGBM model ranks better: test AUC 0.870 vs 0.860, a gap of +0.010 (95% CI 0.007 - 0.013).

| | Scorecard (deployed) | LightGBM (benchmark) |
|---|---|---|
| Ranking (test AUC) | 0.860 | 0.870 |
| Explainability | Points per bin, readable by a credit committee | Needs post-hoc explanation (e.g. SHAP) |
| Reason codes | Exact, from the points table | Approximate |
| Validation effort | Low: 7 features, monotonic bins, signed coefficients | Higher: interactions, tuning, stability |
| Use in this project | Production decisions | Shadow benchmark for monitoring |

**Recommendation.** Deploy the scorecard. The challenger's gap shows how much ranking a more complex model could add. It is the trigger for a future review if a lender's validation team can support an ML model with compliant explanations.

## 7. Costs and risks

- **Good customers declined.** 9.5% of good applicants are turned away at 549 (measured). That is lost revenue and possible complaints. A grey zone around the cutoff could go to manual review instead of automatic decline.
- **Approved-population bias.** The data contains only previously approved borrowers, so the true performance on the full applicant population is unknown until outcomes are observed (see [assumptions.md](assumptions.md), Limitations).
- **Model risk.** Performance will decay as the economy and the applicant mix change. The monitoring thresholds trigger investigation (PSI > 0.10) and action (PSI > 0.25). Both are Benchmark values.
- **Operating costs not modelled.** Infrastructure, model validation, and the compliance review of reason-code wording.

## 8. KPIs to track after launch

| KPI | Target / trigger | Source |
|---|---|---|
| Approval rate | Within 3 points of 87% (investigate), 6 points (act) | Audit log. Thresholds: Assumption |
| Score PSI and feature CSI | < 0.10 stable, > 0.25 act | Monitoring report. Thresholds: Benchmark |
| Early-delinquency rate of approved accounts | Track against the predicted PD by score band | Loan system (outcomes take months to mature) |
| Decision latency and errors | p95 latency; 503 rate (audit write failures) | API logs |
| Reason-code mix | Sudden shifts point to data or upstream changes | Audit log |

## 9. Next steps for a real deployment

1. Replace the Illustrative money inputs with the lender's own funding cost, margin and recovery data, and re-derive the cutoff.
2. Revalidate on the lender's own applicant population, with reject inference or a controlled below-cutoff test.
3. Add authentication, TLS and a retention job (Regulation B: 25 months) before any production traffic.
4. Agree a referral band around the cutoff for manual review.
5. Turn on branch protection so that changes cannot be merged unless CI passes.
