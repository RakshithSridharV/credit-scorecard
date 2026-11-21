# Decision log

Each entry records **what** was decided, **why**, the **alternatives** considered,
and any **open risks**. Evidence for data decisions is in
[reports/data_audit.md](../reports/data_audit.md). Status: *Approved* (by project
owner), *Open* (still to be resolved), or *Required* (work that must happen later).

---

## D-001 Load: drop the row number, enforce the schema
*Approved, 2026-11-18*

- **What.** `load_raw()` drops `Unnamed: 0` and raises if any expected column is missing
  or an unexpected one appears.
- **Why.** `Unnamed: 0` is the CSV row number, not a customer ID, so it carries no
  information. A schema check makes a renamed or missing field fail loudly
  instead of flowing silently into a model.
- **Alternatives.** Keep `Unnamed: 0` as a feature (rejected: it would just encode
  file order). Its value is kept only as the `row_id` index for lineage (D-012).

## D-002 How thresholds were chosen
*Approved, 2026-11-18*

- **What.** All cleaning thresholds below (income 0/1, utilisation > 10, codes 96/98,
  age < 18) were chosen from an audit of the **full** file, before the split.
- **Why this is acceptable.** These rules target placeholder artifacts and impossible
  values, not statistical patterns, so the effect of having seen valid/test rows is
  negligible. They are fixed business rules, not parameters estimated from data.
- **Rule going forward.** Anything that informs a **learned** choice (bin edges, WoE
  values, IV-based feature selection, imputation values, model coefficients, the cutoff)
  must use the **training split only**. Valid is for tuning, test is touched once at the end.

## D-003 `MonthlyIncome`: treat 0 and 1 as missing; no imputation
*Approved, 2026-11-18*

- **What.** Set `MonthlyIncome` to NaN where it is 0 or 1. Existing NaN stays NaN.
  No imputation. The WoE stage gives "Missing" its own bin.
- **Why.** 29,731 rows are missing, 1,634 are 0 and 605 are exactly 1. Their bad rates
  (5.6%, 4.0%, 2.8%) are far below genuinely low earners ($1k-2k: 10.6%), so 0 and 1
  behave like "not reported", not "no income". Median imputation would hide the fact
  that missing income is lower risk.
- **Alternatives.** Median imputation fitted on train (rejected: destroys the
  missingness signal); keep 0/1 as real values (rejected: breaks the "lower income,
  higher risk" pattern in the lowest bin); add an `income_reported_zero` flag
  (rejected for now: small groups with similar risk, and one less feature to explain).

## D-004 `DebtRatio`: NaN where income is missing, 0 or 1
*Approved, 2026-11-18*

- **What.** Set `DebtRatio` to NaN wherever the **original** income is missing, 0 or 1.
- **Why.** In those rows DebtRatio is a dollar amount, not a ratio (median 1,159 vs 0.29;
  100% whole numbers when income is missing). The provider appears to have divided by 1.
  Among DebtRatio > 1 rows the bad rate is 12.8% when income is real but 5.4% when it is
  not: two different populations would share one bin.
- **Alternatives.** Keep as is (rejected: mixes dollars with ratios); keep the amount as a
  separate `MonthlyDebt` feature (rejected: defined for only about 21% of rows).
- **Implementation note.** The rule must give the same result whether it runs before or
  after D-003, so it checks "income is NaN, 0 or 1", not just "income is NaN".

## D-005 `NumberOfDependents`: keep NaN
*Approved, 2026-11-18*

- **What.** Leave missing dependents as NaN (own WoE bin).
- **Why.** 3,924 missing rows, every one of them also missing income: one cause explains
  both. Their bad rate (4.6%) differs from the population, so the gap is informative.
- **Alternatives.** Impute 0 as a fixed rule, or the train mode (rejected: hides the signal).

## D-006 Past-due columns: 96/98 codes become NaN
*Approved, 2026-11-18. Open risks R-1 and R-2 below.*

- **What.** In all three past-due columns, set the values 96 and 98 to NaN.
- **Why.** Real counts stop at 17. The codes appear in 269 rows, always in all three
  columns at once, with utilisation fixed at 0.9999999 and almost no open credit lines:
  placeholders, not counts. Their bad rate is 54.6%, so the rows are kept and the signal
  lives in the Missing bin.
- **Alternatives.** Drop the rows (rejected: removes the riskiest group and makes the
  portfolio look safer); cap at 17 (rejected: pretends to know the count); leave as is
  (rejected: treats 98 as "very late" for the wrong reason).
- **Open risk R-1: possible information from after the outcome.** The dataset does not
  document when or why 96/98 were recorded. If the code reflects an account status set
  *after* the borrower defaulted (e.g. account closed or charged off), the feature leaks
  the target and the 54.6% bad rate would be partly circular. This cannot be checked in
  this data. Mitigation: report model performance with and without these rows at the
  evaluation stage, and list this risk in the model card.
- **Open risk R-2: API behaviour for 96/98.** The scoring API must define what happens
  when a client submits 96 or 98 in a past-due field: either reject the request (422) or
  map the value to NaN exactly as in training. To be decided at the API stage; the choice
  must match training behaviour and be covered by a test.

## D-007 Utilisation: values above 10 become NaN
*Approved, 2026-11-18*

- **What.** Set `RevolvingUtilizationOfUnsecuredLines` to NaN where it is > 10. Values
  from 1 to 10 are kept as genuine over-limit borrowing.
- **Why.** Above 100 there are 223 rows, all whole numbers, with a 4.9% bad rate: dollar
  balances typed into a ratio field. Being 10 times over a card limit is not plausible
  lending. Only 18 rows lie between 10 and 100, so the exact cut-off is immaterial.
- **Alternatives.** Cut-off at 100 (defensible from data alone); cap at 1 (rejected: merges
  40%-bad over-limit borrowers with normal ones); drop the rows.
- **Known limitation.** Some dollar balances may also hide below 10 (a $3 balance looks like
  a ratio of 3). They cannot be told apart in this data.

## D-008 Age: drop age 0, keep ages above 100
*Approved, 2026-11-18*

- **What.** Drop rows with age < 18 (one row, age 0). Keep the 13 rows aged 101-109.
- **Why.** No row is aged 1-17, so age 0 is an entry error, and a credit contract needs an
  adult. The API will reject age < 18, so the model never needs to score such a row.
  Ages above 100 are rare but possible; binning merges the tail.
- **Alternatives.** Set to NaN or impute (rejected: invents a value for an invalid record).
- **Note.** Age is sensitive under ECOA / Regulation B (applicants aged 62+ must not be
  scored worse because of age). To be checked at the modelling stage.

## D-009 Duplicates: keep, and measure their effect
*Approved, 2026-11-18. Required evaluation step.*

- **What.** Keep the 609 exact duplicate rows.
- **Why.** There is no customer ID. The repeats are mostly thin-file profiles where many
  different people share identical values, and 145 rows have identical features but
  different outcomes. Dropping them would under-represent a real segment.
- **Alternatives.** Drop exact duplicates before the split (also defensible).
- **Required later.** At the evaluation stage, compute AUC on **valid and test** with and
  without duplicates and report the difference. Proposed definition of "without
  duplicates": remove valid/test rows whose feature values exactly match a train row,
  and keep one copy of duplicates within the split. To be confirmed at that stage.

## D-010 Long tails: no capping
*Approved, 2026-11-18*

- **What.** No winsorising of income, credit lines, real-estate loans or dependents.
- **Why.** The values are rare but possible. WoE binning is robust to tails, and the
  LightGBM challenger is unaffected by them.
- **Alternatives.** Cap at train 1st/99th percentiles (would be a learned step, fit on
  train only); not needed for a binned model.

## D-011 Class imbalance: no resampling
*Approved, 2026-11-18*

- **What.** Keep the natural 6.68% bad rate. No oversampling, SMOTE or class weights for
  the scorecard.
- **Why.** The score maps to a probability of default used for pricing and provisioning,
  so it must reflect the true bad rate. Imbalance is handled by stratifying the split
  and judging with AUC / Gini / KS rather than accuracy.
- **Alternatives.** SMOTE or oversampling (rejected: distorts PDs unless corrected);
  class weights (may be tried for the challenger only).

## D-012 Split: 70/15/15, stratified on the target, seed 42
*Approved, 2026-11-18*

- **What.** `python -m scorecard.data.split` splits the raw data into train / valid / test
  (70/15/15), stratified on `SeriousDlqin2yrs`, `seed: 42` from `configs/config.yaml`.
  Files are saved to `data/processed/` with the original row number as `row_id` (lineage
  only, never a feature). `data/raw` is read, never written.
- **Why.** Splitting before cleaning or fitting guarantees nothing learned can see valid
  or test rows. Stratifying keeps the bad rate identical across splits, which matters
  with only about 1,500 bads in each of valid and test. Valid is for choices (bins,
  features, cutoff); test is used once for the final report.
- **Alternatives.** Out-of-time split (best practice for scorecards, impossible here: the
  data has no dates, so this is a stated limitation); 60/20/20 (more evaluation data,
  less training data); k-fold cross-validation (useful later for model selection on train).
- **Known limitation.** A random split only tests whether the model generalises to new
  borrowers from the *same period*. It cannot show stability over time.

## D-013 Cleaning: one pure function, row filtering kept separate
*Approved, 2026-11-18*

- **What.** `python -m scorecard.data.clean` reads each split from `data/processed/`,
  applies `drop_invalid_rows()` (D-008) then `clean()` (D-003, D-004, D-006, D-007), and
  writes `data/interim/{train,valid,test}.csv`. Counts are in
  [reports/clean_summary.md](../reports/clean_summary.md).
- **Why.**
  - `clean()` learns nothing from data, so running it on every split cannot leak.
  - It needs no target column and never drops rows, so the scoring API can call the
    exact same function. This avoids training/serving skew.
  - The "income not reported" mask is computed once from the input and shared by D-003
    and D-004. Rule order can't change the result, and `clean(clean(x)) == clean(x)`
    (covered by a test).
  - Row dropping lives in `drop_invalid_rows()` because the API must reject an
    under-18 applicant with an error, not silently drop them.
- **Result.** The single age-0 row was in train (105,000 → 104,999). In every split
  `MonthlyIncome` and `DebtRatio` now have identical NaN counts (train 22,332, 21.3%),
  as D-004 intends.
- **Alternatives.** An sklearn `Pipeline` transformer (not needed while no step is fitted;
  can wrap `clean()` later); thresholds in YAML (rejected: they are fixed business rules
  tied to decision IDs, not run settings, so they live as named constants in code).

## D-014 Binning and WoE: own implementation, monotonic, 2% floor, 6-bin cap
*Approved, 2026-11-20*

- **What.** `python -m scorecard.features.encoder` fits a `WoEEncoder` on **train only**
  and saves it to `models/woe_encoder.json`; IV and bin tables are in
  [reports/woe_summary.md](../reports/woe_summary.md). Per feature:
  1. Fine classing: 20 quantile bins (`inverted_cdf`, so edges are real data values).
  2. Merge any bin below **2%** of train rows into the neighbour with the closer bad rate.
  3. Merge until the bad rate is strictly monotonic (pool-adjacent-violators), in both
     directions.
  4. Merge the least informative cut until at most **6** numeric bins remain.
  5. Keep the direction with the higher IV. NaN is always its own Missing bin and is
     never merged. WoE = ln(%good / %bad) with 0.5 added to each bin's counts; an empty
     bin (e.g. Missing for a feature with no NaN in train) gets WoE 0.
- **Why.**
  - *Monotonic:* every score step must be explainable ("more utilisation, fewer points");
    a zig-zag would be fitted noise and is hard to defend to a regulator.
  - *2% floor, not the usual 5%:* only 4.9% of train rows have any 60-89-day lateness,
    with a 35.6% bad rate against 5.1% for the rest. A 5% floor merges them away and the
    feature disappears. 2% is still about 2,100 rows per bin.
  - *6-bin cap:* uncapped, age had 17 bins and utilisation 14; neighbouring WoEs differed
    by about 0.03 and the order broke 15 times on valid. Tested on valid (allowed by
    D-002): cap 6 cuts that to 3 breaks for a 1.7% loss of total valid IV (4.01 to 3.94).
    Caps 8 and 5 were also tried (5 breaks / 2 breaks).
  - *JSON artifact, not pickle:* readable, diffable, and safe to load in the API.
- **Result.** Highest IV: utilisation 1.13, 90+ days late 0.84, 30-59 days 0.74,
  60-89 days 0.57, age 0.25. Lowest: dependents 0.036, real-estate loans 0.044.
  Valid IV is close to train for every feature, so the bins did not overfit.
- **IV > 0.5 flag.** The four features above 0.5 are payment-behaviour variables, which
  are expected to be strong. The 96/98 Missing bin adds only 0.04 to each past-due IV,
  so their strength does not come from the R-1 codes. The flag is noted, not acted on.
- **Alternatives.** `optbinning` (rejected: another dependency to clear through
  Application Control, and a solver is harder to defend line by line); ChiMerge
  (merging by a chi-square test; similar result, one more parameter); decision-tree
  binning (fast but not monotonic by default); equal-width bins (empty bins on long tails).
- **For later steps.** The three past-due features share one Missing bin (the same 183
  rows) and are correlated: check this at feature selection. Real-estate loans and open
  credit lines lose a third of their IV on valid (0.044 to 0.025, 0.084 to 0.056), so they
  are candidates to drop. Age 62+ gets positive WoE (scored better), consistent with
  the D-008 Regulation B note.

## D-015 Feature selection: 7 of 10 features, chosen by CV on train
*Approved, 2026-11-20*

- **What.** `python -m scorecard.features.selection` applies three gates on **train only**
  and saves the result to `models/selected_features.json`. Details in
  [reports/feature_selection.md](../reports/feature_selection.md).
  1. IV >= 0.02. 2. Pairwise WoE correlation <= 0.7, then VIF <= 5 (the weaker feature goes).
  3. Backward elimination by 5-fold stratified CV AUC (seed 42), WoE refitted inside each
     fold, unpenalised logistic regression. Each round removes a wrong-sign feature first;
     otherwise the cheapest feature if dropping it costs < 0.0005 AUC.
- **Result.** Kept: utilisation, 90+ days late, 30-59 days, 60-89 days, DebtRatio, age,
  real-estate loans. Dropped:
  - `MonthlyIncome`: coefficient +0.055 (wrong sign: "more income, more risk"). DebtRatio is
    debt / income and has exactly the same Missing rows, so it already carries this signal.
  - `NumberOfDependents` (AUC loss 0.00003) and `NumberOfOpenCreditLinesAndLoans` (0.00008):
    within fold noise, not consistently worse across folds.
  Gates 1-2 removed nothing: all IVs >= 0.036, max WoE correlation 0.33, max VIF 1.51.
  CV AUC 0.8545 with 10 features, 0.8545 with 7. Valid AUC (reported only, not used to
  choose): 0.8587 with all 10, 0.8592 with the 7.
- **Why.**
  - A wrong-sign coefficient cannot be defended to a regulator or an applicant.
  - Fewer features means fewer data fields to collect and monitor, and fewer reasons to
    explain on a decline. `NumberOfDependents` also touches family status, an area fair-lending
    reviewers look at closely.
  - Real-estate loans stays despite its low IV: dropping it costs 0.0011 AUC, worse in 5 of
    5 folds. It works with DebtRatio (mortgage holders have high debt ratios but repay well).
  - The 0.0005 tolerance sits in a wide gap: dropped features cost <= 0.00012, the weakest
    kept one costs 0.0008-0.0011. Any value in between gives the same set.
  - Using CV on train, not valid, keeps valid an honest check for the model step.
- **Alternatives.** Select by valid AUC (rejected: valid would be used both to choose and to
  judge); forward selection (same answer here, less common in scorecard practice); L1/lasso
  (picks features automatically but gives shrunk coefficients that are harder to explain);
  keep all 10 (rejected: a wrong-sign coefficient).
- **Note.** Income is still needed upstream, because the bureau computes DebtRatio from it.

## D-016 Scorecard model: unpenalised WoE logistic regression, 600/50/20 scale, isotonic master scale
*Approved, 2026-11-20. Open item (96/98 rows) carried to the test evaluation.*

- **What.** `python -m scorecard.models.scorecard` fits the scorecard on **train** with the 7
  D-015 features and saves `models/scorecard.json` (encoder, coefficients, integer points per
  bin, score-to-PD table: one self-contained file for the API). Evidence:
  [reports/scorecard_model.md](../reports/scorecard_model.md).
  - Model: logistic regression on WoE, no penalty. All 7 coefficients negative, |z| >= 6.9.
  - Scale (**Assumption: industry convention**, Siddiqi): 600 points = 50:1 good:bad odds,
    20 points to double the odds. Points per bin = -(b_i * WoE + b0/n) * factor + offset/n,
    rounded to integers. Valid scores run from 416 to 627, median 593.
  - PD: a score-to-PD table ("master scale") fitted by isotonic regression on **out-of-fold**
    train scores (5 folds, seed 42), never decreasing in risk as the score falls. PD floor
    0.05% (**Benchmark:** Basel III minimum PD for retail exposures).
- **Result on valid** (test untouched): AUC 0.859, Gini 0.718, KS 0.562. Rounding to integer
  points costs 0.0001 AUC. Calibration by score decile: predicted/observed ratio 0.80-2.16 on
  the raw scale, 0.88-1.12 after the master scale; Brier 0.05081 -> 0.05007, log loss
  0.1840 -> 0.1811; mean PD / bad rate 0.99.
- **Why.**
  - *Variants tried* (CV AUC on train / valid AUC): 7 WoE features 0.8545 / 0.8592; all 10
    0.8545 / 0.8587; L2 C=0.01 0.8541 / 0.8594 and C=0.001 0.8522 / 0.8580 (a penalty adds
    nothing at 105k rows); raw features without WoE 0.8497 / 0.8585 (worse on CV and not a
    points scorecard).
  - *Class weights rejected* (D-011): AUC 0.8607 but mean PD 33.6% against a 6.7% bad rate.
    Even after recalibration its gain over the chosen model is 0.0016 AUC, not worth
    "upweight bads 14x, then undo it" in a model that must be explained.
  - *Why a master scale:* the raw scale over-predicted the safest decile about 2x (0.96% vs
    0.44%) and under-predicted deciles 7-9. The same pattern appears in out-of-fold train
    predictions, so it is model shape, not noise. Platt scaling (a linear logit map) did
    nothing (slope 0.998): logistic regression is already calibrated on average, and the
    error is curvature. Isotonic fixes curvature and keeps the points additive.
- **Alternatives.** Platt scaling (no effect, above); isotonic on the exact PD instead of the
  score (same result, but two applicants with the same score could get different PDs);
  refit on train + valid before deployment (to decide after the test evaluation).
- **Open.** The 183 rows with 96/98 codes take the Missing points in all three past-due
  features (about 135 points below a clean record). Consistent with their 56% bad rate, but
  this is where risk R-1 bites: test with and without them at evaluation.

## D-017 Challenger: LightGBM benchmark, the scorecard stays the production model
*Approved, 2026-11-20. Challenger kept for shadow scoring in monitoring.*

- **What.** `python -m scorecard.models.challenger` tunes LightGBM by 5-fold CV on **train**
  (the scorecard's folds, seed 42), compares it with the deployed scorecard PD on **valid**,
  and saves the monotone variant to `models/challenger_lgbm.txt` as a benchmark only. Evidence:
  [reports/challenger.md](../reports/challenger.md). Test untouched.
  - Grid: num_leaves {7, 15, 31} x min_child_samples {100, 400}; learning rate 0.05; trees
    chosen by early stopping (100 rounds); 80% feature and row subsampling; L2 = 1.
  - Three variants, to split the gap by cause: 7 scorecard features; all 10; all 10 with
    monotone constraints set from credit logic (open lines and real-estate loans left free,
    known U-shapes).
- **Result on valid** (valid AUC, gap over the scorecard's 0.8592 with paired-bootstrap 95% CI,
  1,000 resamples):
  - 7 features: 0.8688, +0.0096 [0.0065, 0.0125]: model flexibility alone.
  - All 10: 0.8707, +0.0115 [0.0082, 0.0147]: the 3 dropped features add about 0.002.
  - All 10, monotone: 0.8694, +0.0102 [0.0074, 0.0130]: constraints cost about 0.001.
  - The gap is real (every resample positive) and about 1 AUC point (Gini +0.02, KS +0.02).
    CV on train agrees (scorecard 0.8545, LightGBM all 10 0.8640), so it is not valid-set
    luck. Brier 0.0501 -> 0.0487-0.0493.
  - The unconstrained model breaks credit logic: for all 200 sampled valid rows, PD moves
    the wrong way somewhere as utilisation, age, debt ratio or income rise, by up to 6.1,
    10.9, 6.3 and 13.7 PD points. Real reversals, not rounding. The monotone variant has none.
- **Why the scorecard stays.** A 0.01 AUC gain does not outweigh what the scorecard gives:
  every point traceable to one bin, reason codes read straight off the points (needed for US
  adverse-action notices, ECOA/Regulation B), a table a credit committee can sign off, and
  no software dependency at scoring time. The challenger's value is in setting the ceiling:
  the scorecard keeps 97% of the best Gini found (0.7184 / 0.7414). Gini, not AUC, for the
  ratio: AUC starts at 0.5, so an AUC ratio (99%) makes any gap look smaller.
  - *Where the gain comes from:* gain shares are the same top 4 features as the scorecard
    (90-day, utilisation, 30-59, 60-89: about 86%). LightGBM uses them more finely (no 2%
    bin floor, interactions), not different information.
- **Alternatives.** XGBoost/CatBoost (same model family, similar AUC expected on 10 tabular
  features; LightGBM already pinned and runs under Application Control); a larger
  random/Bayesian search (the 6 grid points differ by at most 0.0011 CV AUC, a tenth of
  the gap we are measuring); DeLong test for the AUC difference (equivalent answer; the bootstrap is
  easier to explain and also gives an interval); deploying the monotone challenger with SHAP
  reason codes (rejected for this project: explanations become estimates, not a lookup).
- **Open.** Keep the challenger for **shadow scoring** in monitoring: if the gap widens on new
  data, the scorecard is missing a pattern and is due for redevelopment.

## D-018 Cutoff: approve when calibrated PD <= break-even PD (score 549 under base assumptions)
*Approved, 2026-11-20. Limitations (reject inference, simplified money) stated in the report.*

- **What.** `python -m scorecard.evaluation.cutoff` sets the approve/decline cutoff from the
  master-scale PD (D-016, fitted on train) and checks it on **valid** outcomes. Evidence:
  [reports/cutoff_strategy.md](../reports/cutoff_strategy.md). Test untouched.
  - Rule: approve if PD <= break-even PD = profit per good / (profit per good + loss per bad).
    The master scale never rises with the score, so the rule is a single score cutoff.
  - Money inputs (config `cutoff:`): exposure $5,000 (**Illustrative**); yearly margin 5% after
    funding and servicing (**Assumption**); 2-year horizon (data fact: the target window);
    LGD 75% (**Assumption**). Profit per good $500, loss per bad $3,750, break-even PD 11.76%
    (all **Illustrative**, derived).
- **Result on valid.** Cutoff **549**: approves 86.9%, bad rate among approved 3.04% (vs 6.68%
  overall), keeps out 60.5% of bads while declining 9.8% of goods. Profit $3.22M vs $2.16M per
  10,000 applicants for approve-all: **+$1.06M per 10,000 (Illustrative)**.
  - Check: the best cutoff found directly on valid is 552, 3 points away and 0.4% more profit,
    which is sample noise. Marginal bad rates agree: 14.8-15.4% in the 10 points below 549
    (above break-even, decline), 8.3-10.5% in the 10 points above (below, approve).
  - Top 80% for reference: cutoff 559, bad rate among approved 2.51%.
  - Sensitivity: loss per bad halved -> cutoff 532, approve 92.2%; doubled -> 569, approve
    72.6%. The cutoff moves about 20 points (one PDO) either way: the assumption matters more
    than any model choice since D-016.
- **Why.**
  - *Marginal, not average:* a cutoff decision adds or removes applicants at the margin, so their
    bad rate must meet break-even. The average bad rate among approved (3.04%) is far below it
    and would wrongly suggest approving much deeper.
  - *Rule from train, check on valid:* maximising profit on valid would fit its noise (and
    using valid both to choose and to report makes the reported profit optimistic). The PD
    rule uses only train-fitted pieces plus business inputs.
- **Alternatives.** Target approval rate (e.g. "approve 80%": simple, but ignores economics);
  target bad rate among approved (same issue, uses the average not the margin); maximise
  profit on valid (noise, above); a grey zone with manual review between two cutoffs (useful
  in practice; needs review costs we do not have); risk-based pricing instead of one cutoff
  (out of scope: no price data).
- **Open.**
  - *Reject inference.* GMSC outcomes exist only for people who were given credit. Applicants
    a past policy declined are missing, so bad rates at low scores are likely understated for
    a true through-the-door population. Not fixable with this data; stated as a limitation.
  - *Simplified money:* no income earned before default, no discounting, no acquisition cost,
    one exposure for all. Fine for a method demo, not for setting a real lending policy.

## D-019 Reason codes: points lost against the train-average applicant, >= 5 points, at most 4
*Approved, 2026-11-20. Age stays in the model and is disclosed when it is a principal reason (owner may revisit).*

- **What.** `python -m scorecard.explain.reason_codes` gives each declined applicant up to 4
  adverse-action reasons (ECOA / Regulation B) read off the points table. Reference points are
  learned on **train** and saved to `models/reason_reference.json` for the API. Evidence:
  [reports/reason_codes.md](../reports/reason_codes.md), on valid declines at cutoff 549 (2,958
  applicants, 13.1%). Test untouched.
  - Points lost on a feature = average points for that feature on train - points received.
  - A reason must cost >= 5 points (**Assumption**: a quarter of the 20-point PDO, about 19%
    higher odds of bad). At most 4 (Regulation B commentary: more is unlikely to help). The
    largest loss is always given, so no declined applicant gets an empty notice.
  - One code and plain text per feature, plus a separate "could not be verified / not reported"
    text for Missing. The three past-due features share PAY_UNKNOWN, given once.
- **Result on valid declines.** First reason: utilisation 48.8%, 90+ days late 31.9%, 30-59 days
  13.5%, 60-89 days 3.3%, payment history unknown 1.6%, debt ratio 1.0%. Reasons per applicant:
  1 (2.3%), 2 (41.7%), 3 (38.5%), 4 (17.5%). Regulation B age check (12 CFR 1002.6(b)(2)):
  applicants 62+ get at least 85 age points, equal to the best under-62 group. **Pass.**
- **Why.**
  - *Average, not best bin.* First built with the best bin as reference: age was in the top 4
    for 85% of declines, because anyone under 63 "loses" points for not being 68+, and trivial
    2-point losses became reasons. Against the average, a reason means "this pulled you below a
    typical applicant", which is what a principal reason for the decline is.
  - *A minimum loss.* A reason should matter to the decision. 5 points is tied to the score
    scale, not tuned to the output.
  - *Exact, not estimated.* The same table that makes the score makes the reasons, so they
    cannot disagree with the decision (unlike SHAP on the challenger, D-017).
- **Alternatives.** Best-bin reference (above; it does answer "what would most raise my score",
  which suits a customer-education screen, not the notice); reference at the cutoff score;
  no threshold (gives 2-point reasons); reasons by WoE rank (ignores the coefficients, so it
  can disagree with the points).
- **Open.**
  - *Threshold sensitivity on age.* The (33, 46] age bin loses 4.9 points against the average, just
    under the threshold. Age appears in the top 4 for 19.8% of declines at >= 5 points, and about
    49% if the threshold were 4. The rule was set from the PDO, but a compliance reviewer should
    know the result sits on this edge.
  - *Age as a disclosed reason.* Age is in the model (D-015) and passes the 62+ rule. If it is a
    principal reason it must be disclosed: hiding it would make the notice inaccurate. If the
    business does not want age-based declines, the remedy is removing age from the model, not
    from the notice. Owner decision; this is a US-data method demo, not legal advice.

## D-020 Test evaluation: performance holds; deploy the train-fitted model as tested, no refit
*Approved, 2026-11-20. Train-fitted model ships as fingerprinted in the lock.*

- **What.** `python -m scorecard.pipelines.evaluate` scored the untouched **test** set once, at
  2026-11-20 13:46 UTC, with all artifacts frozen. The lock
  [reports/test_evaluation.lock.json](../reports/test_evaluation.lock.json) holds the SHA-256 of
  the scorecard, challenger, reason reference, config and test file; a second run is refused
  without `--force` and would be logged. The pipeline was first rehearsed on valid
  (`--rehearse`, writes nothing), which reproduced every earlier valid number. Evidence:
  [reports/test_evaluation.md](../reports/test_evaluation.md).
- **Result on test** (valid in brackets):
  - AUC **0.860** [0.859], 95% CI 0.851-0.870; Gini 0.720 [0.718]; KS 0.569 [0.562].
    CV on train 0.855 -> valid 0.859 -> test 0.860: no sign the choices made on valid overfit it.
  - Calibration: mean PD / bad rate 0.98; deciles 4-10 within 0.93-1.11. Deciles 2-3 over-predict
    (0.68% vs 0.49%, 1.09% vs 0.76%), but each rests on 11-17 bads, within noise, and errs on
    the safe side.
  - Policy at cutoff 549: approve 87.0% [86.9%], bad rate among approved 2.97% [3.04%], bads
    kept out 61.4% [60.5%]. Value over approve-all **$1.10M per 10,000 applicants
    (Illustrative)** [$1.06M].
  - Challenger gap: +0.0101 AUC [+0.0102], CI 0.0072-0.0130. The D-017 conclusion holds.
  - Reason codes on test declines: utilisation first for 52.0%, 90+ days late 30.4%; age in the
    top 4 for 21.2% [19.8%].
- **Risk R-1 (96/98 rows), mitigation done.** 40 test rows, bad rate 55.0%, all declined.
  Without them AUC is 0.859 instead of 0.860: the model does not rely on them for its ranking,
  so even if the codes were recorded after default (cannot be checked), the effect on reported
  performance is about 0.001 AUC. Their PD (64.4%) is above the observed rate (55.0% test, 47.8%
  valid; small groups), which does not change the decision since break-even is 11.8%. R-1 stays
  listed in the model card as an unverifiable data-lineage risk.
- **Why no refit on train + valid.** A refit re-learns bins, WoE, points and the master scale,
  so it is a different model with no untouched data left to test it. The gain would be about
  21% more rows (105k -> 127.5k), small next to the stable CV -> valid -> test results above.
  The deployed model is the one the lock fingerprints: what was tested is what ships.
- **Alternatives.** Refit on train + valid (or all data) with the specification frozen: common
  in industry and usually slightly better, but needs a fresh out-of-time sample to validate,
  which this dataset does not have; recalibrate deciles 2-3 (rejected: within noise, and
  changing anything after looking at test spends the test set).

## D-021 Monitoring: PSI + CSI + approval drift against a frozen train baseline; worst signal wins
*Approved, 2026-11-20. Monitoring = worst of score PSI, CSI and approval drift.*

- **What.** `python -m scorecard.evaluation.psi` freezes a **train** baseline in
  `models/monitoring_baseline.json` (score decile edges and shares, each feature's bin shares,
  mean points), compares populations with it, and runs simulated drift on valid. No model
  changes. Evidence: [reports/monitoring.md](../reports/monitoring.md).
  - Score PSI on train-decile bands; CSI per feature on the scorecard's own bins (the bins that
    move points), plus the mean points shift per feature; empty bands floored at 0.01%.
  - Thresholds: PSI/CSI < 0.10 stable, 0.10-0.25 investigate, > 0.25 act (**Benchmark**:
    industry rule of thumb, Siddiqi). Approval-rate drift from train's 87.0%: > 3 points
    investigate, > 6 points act (**Assumption**). OVERALL = the worst of the three.
- **Result.** Real: train vs valid PSI 0.0004, train vs test 0.0009, all CSI < 0.001: stable,
  as expected for random splits of one population. Simulated (**Illustrative**, on valid):
  - Utilisation +30%: score PSI **0.045 (stable)**, utilisation CSI **0.41**, approval 87.0% ->
    78.5%: OVERALL **ACT**.
  - Under-35s apply 2x: PSI 0.008, age CSI 0.06, approval 85.5%: stable; CSI names age.
  - 10% gain a 30-59 late payment: PSI 0.005, 30-59 CSI 0.05, approval 85.5%: stable; CSI names it.
- **Why.**
  - *Score PSI alone is not enough.* In the downturn scenario the score fell 5 points on average
    (a quarter of a PDO) and approvals dropped 8.5 points, but decile shares barely moved, so
    score PSI stayed under 0.10. CSI and the approval rate caught it. Hence three signals.
  - *Frozen baseline.* A monitoring job months later must compare against the same bands and
    must not need the training data.
  - *CSI on the scorecard's bins,* not fresh quantiles: a shift inside one bin does not change
    any score, so it should not raise an alarm.
- **Alternatives.** Score PSI only (misses the downturn, above); 20 score bands (more sensitive,
  noisier at month-level volumes); KS or Wasserstein distance between score distributions (no
  shared industry thresholds); per-feature drift with statistical tests (with large volumes
  every tiny shift is "significant").
- **Plan.** Monthly: score PSI, CSI, approval rate, input data checks. Quarterly once outcomes
  mature (2-year window): calibration ratio, Gini on matured accounts, challenger gap (D-017).
  Thresholds beyond PSI/CSI are **Assumptions**; full table in the report.
- **Note on the test lock (D-020).** `configs/config.yaml` was edited after the test evaluation
  to add the `monitoring:` section and two output paths, so its SHA-256 no longer matches the
  lock. No section used by the model, cutoff or reason codes changed.

## D-022 Training pipeline: one command from raw CSV, verified byte-identical to the tested model
*Approved, 2026-11-21. Rebuild is verified against the first test-lock run.*

- **What.** `python -m scorecard.pipelines.train` (or `make train`) runs the 10 stages in order
  from `data/raw/cs-training.csv`: audit, split, clean, WoE encoder, feature selection,
  scorecard, challenger, cutoff report, reason reference, monitoring baseline. It then compares
  the SHA-256 of the rebuilt scorecard, challenger, reason reference and cleaned test file with
  the **first** run in the test lock (D-020) and exits with an error if any differs.
  `--no-verify` skips the check for a deliberate redevelopment. The test evaluation is not a
  stage: training never scores test. A `Makefile` gives `train`, `rebuild`, `rehearse`,
  `evaluate`, `test`, `lint`, `check`.
- **Result.** Full rebuild in 242 s (challenger tuning 166 s, feature selection 60 s). All four
  locked artifacts **byte-identical** to the tested ones. Every other file in `models/`,
  `reports/`, `data/processed/` and `data/interim/` was also identical to a backup taken first.
- **Why.**
  - *Auditability.* "Can you rebuild the model you tested?" is now one command with a pass/fail
    answer, not a list of steps to remember.
  - *Why it is byte-identical:* fixed seeds everywhere (split, CV folds, bootstrap); LightGBM
    `deterministic=True` and `force_row_wise=True`; scikit-learn's solvers are deterministic for
    the same data and thread count; JSON written with a fixed key order and float formatting.
  - *Compare with the first lock run,* not the latest: a forced re-run must not quietly become
    the reference.
- **Alternatives.** A workflow tool (DVC, Airflow, Prefect: caching and scheduling, but extra
  dependencies and setup for a 4-minute, 10-stage build); stage-level caching (adds staleness
  bugs for little gain at this size); hashing only the scorecard (the API also loads the
  reason reference, and the challenger is the monitoring benchmark).
- **Limit.** Byte-identical holds on this machine and these pinned versions. Another OS, CPU or
  library version may change floating-point results in the last digits. The check would then
  fail loudly, which is the right behaviour: re-validate before shipping.

## D-023 Scoring core: one web-free module, same `clean()`, verified artifacts, refuse what was never trained on
*Approved, 2026-11-21. Decline-only reasons; income-withholding risk noted for policy.*

- **What.** `src/scorecard/api/scoring.py`. `ScoringService.from_config()` loads
  `models/scorecard.json` and `models/reason_reference.json` once. `score(applicant)` and
  `score_batch(rows)` return a `Decision`: integer score, calibrated PD (master scale), approve or
  decline at the D-018 cutoff (549), up to 4 reason codes **for declines only**, the cutoff used,
  and `model_version` (first 12 hex of the scorecard's SHA-256). No FastAPI import: the API,
  Streamlit and batch jobs all call this module.
- **Skew guards.**
  1. Inputs go through the **same `clean()`** as training, not a copy of it.
  2. At start-up the two files the decision depends on are hashed and compared with the **first**
     test-lock run (D-020). A mismatch or a missing lock stops the service (`ArtifactMismatch`).
  3. An uncalibrated scorecard, or a reason reference for other features, is refused.
- **Input rules.** All 10 raw fields are accepted, including the 3 the model does not score:
  `MonthlyIncome` is needed because `clean()` uses it to decide whether `DebtRatio` is a ratio
  (D-004). `MonthlyIncome` and `NumberOfDependents` may be null or absent (missing in training
  too). Every other field is required. Refused, with **every** problem listed: unknown field
  names (a typo like `MonthlyIncom` would otherwise score as "income not reported"), non-numbers,
  inf, negatives, non-whole counts, **age under 18** (training dropped these rows, D-008; live
  they are refused, not scored). Past-due codes 96/98 are accepted and become missing, as in
  training. A batch is all-or-nothing.
- **Result.** 26 new tests (23 unit, 3 integration), full suite 146 passed, ruff clean.
  **Skew test:** the service on the raw valid rows produces exactly the same score as the
  offline scorecard on the cleaned valid file, for every row.
- **Finding (worked example).** Two applicants differ only in income, and both have
  DebtRatio 0.3. Income 0 gives **589**; income 4000 gives **586**. With income 0, `DebtRatio`
  becomes missing, and the Missing bin (train bad rate 5.4%) scores 3 points above the
  0.02-0.43 bin (6.1%). Both are approved.
- **Why.** One decision path means one thing to validate. Refusing (HTTP 422 later) is safer than
  scoring an input outside the training population. `model_version` in every decision makes each
  one traceable to the exact tested file (needed for the audit log in the DB step). Reasons are
  given only for declines because Regulation B requires them for adverse action. An approval needs
  none, and showing reasons on approvals invites "why not a higher limit?" disputes the model
  cannot answer.
- **Alternatives.** Validation only in the HTTP schema (Pydantic): batch and Streamlit callers
  would bypass it, so the core checks again (defence in depth). Pickling a sklearn pipeline:
  version-fragile and unsafe to load. JSON with a schema version is readable and diffable.
  Storing the cutoff in the model file: the cutoff is a business policy on top of the model, so
  it is recomputed from config and returned in every response. Clipping age < 18 to 18: this
  silently scores an impossible applicant.
- **Open risks.**
  - *Income withholding.* Not reporting income earns +3 points here (above). The effect is small,
    but a real lender should require or verify income (policy, not model).
  - *Config drift.* The cutoff comes from `config.yaml`, which is not hash-checked at start-up,
    so a changed EAD or LGD changes decisions. Mitigated because config lives in git (code review)
    and the cutoff is returned in every decision. Full mitigation would be pinning the expected
    cutoff in the lock.

## D-024 HTTP API: FastAPI, strict schema, load once and fail fast, no applicant data in errors
*Approved, 2026-11-21. Strict contract, fail-fast start-up, no input echo in errors.*

- **What.** `scorecard.api.main:app` (run with `make serve`) has two endpoints.
  `POST /score` takes one applicant's 10 raw bureau fields and returns the `Decision` from the
  scoring core (D-023). `GET /health` returns `status`, `model_version` and `cutoff`. The
  `create_app(service_factory)` factory loads the model once, in the lifespan hook. FastAPI,
  pydantic and uvicorn are pinned in a separate `[api]` extra, so training never installs them.
- **Contract (`schemas.py`).**
  - Wire names are the bureau names (no renaming layer to get wrong).
  - The schema is **strict**. `"45"` (string), `true` (boolean), 1.5 late payments, negatives,
    Infinity/NaN, unknown fields and age under 18 all get a **422** naming the field.
  - Integers are accepted for amount fields, and `MonthlyIncome` and `NumberOfDependents` may be
    null or omitted.
  - The OpenAPI document at `/docs` is the published contract, and a test checks that it lists
    exactly the 10 bureau fields.
- **Fail fast.** If the model fails to load (an `ArtifactMismatch`, a missing file), start-up
  raises and the process exits. An instance that answers is therefore always ready, so one
  `/health` serves as both liveness and readiness.
- **Errors never echo input.** FastAPI's default 422 copies each bad value into the response.
  We replaced it with `{loc, msg, type}`, and core error messages dropped the value too. Two
  reasons: applicant data must not leak into responses or logs, and **found by a test**,
  an echoed `Infinity` cannot be JSON-encoded, which turned a correct 422 into a **500**. Core
  rejections (`InvalidApplicant`) use the same shape, so a client handles one error format.
- **`def`, not `async def`, for `/score`.** Scoring is CPU work (pandas). FastAPI runs sync
  endpoints in a thread pool, so one request never blocks the event loop.
- **Result.** 18 API tests. The full suite has 164 passed, ruff is clean. A smoke test on a real
  uvicorn server with the real, hash-verified model gave `/health` = model `27f4de0069b8`,
  cutoff 549, and about **16 ms per request** (sequential, including HTTP, on the laptop).
- **Alternatives.**
  - *Flask:* no built-in validation or OpenAPI.
  - *Lenient schema with coercion:* "convenient" until a "1,500" income string is parsed wrongly.
  - *Load the model per request:* slow, and two requests could be scored by different files.
  - *Start in a "not ready" state and answer 503:* workable on Kubernetes, but it adds state.
    Fail-fast is simpler, and a rolling deployment keeps the old pods serving.
  - *snake_case wire names:* nicer JSON, but they add a mapping layer that could drift from the
    training columns.
- **Open / next.** No authentication or rate limiting yet (a gateway's job in a real bank; noted
  for the Docker step). No batch endpoint (the core supports batches; an HTTP version needs a size
  cap). Decisions are not yet stored: the audit log is the DB step. The Starlette TestClient warns
  that `httpx` is deprecated in favour of `httpx2`. This is harmless at the pinned versions, and
  changing it was deferred.

## D-025 Decision audit log: append-only table, fail closed, schema changes only by migration
*Approved, 2026-11-21. Fail closed; append-only; schema changes only by migration.*

- **What.** Every issued decision is stored in a `decisions` table: a UUID `id`, `created_at`
  (UTC), `model_version`, `cutoff`, `score`, `pd`, `decision`, `reasons` (JSON) and the 10 raw
  `inputs` (JSON). `POST /score` now returns `decision_id`. The code is in `api/db/`:
  - `models.py` defines the table, with CHECK constraints on `decision` and `pd` and a naming
    convention for constraints.
  - `repository.py` has `record`, `get`, `count` and `ping`, and **no update or delete**.
  - `session.py` holds the URL, the engine and the start-up schema check.
  Alembic migrations live in `migrations/` (`0001` creates the table) and run with `make migrate`.
- **Fail closed.** If the write fails, the response is **503** with `Retry-After: 5`, and **no
  decision** is returned. A decision that cannot be recorded is not issued (Regulation B
  record-keeping, dispute handling). Requests rejected with 422 are not decisions and are not
  stored.
- **Probes split.** `/health` (liveness) checks that the process and model are up. `/ready`
  (readiness) checks that the database answers. A database outage makes the instance not-ready
  (traffic stops) but still alive (no restart): restarting every pod cannot fix a database.
- **Schema only by migration.** The app never creates tables. At start-up it checks that the
  database is reachable and at the latest Alembic revision, and otherwise fails fast
  (`SchemaNotCurrent`: "Run `make migrate`"). Schema changes are reviewed scripts with
  `upgrade`/`downgrade`, and `alembic upgrade head --sql` prints the DDL for a DBA to review.
- **Configuration.** `database.url` in `config.yaml` (local SQLite under the git-ignored
  `data/audit/`) is overridden by the `SCORECARD_DB_URL` environment variable (Postgres in
  Docker), so credentials never live in git. The same SQLAlchemy code runs on both.
- **Personal data.** `hide_parameters=True` keeps SQL parameters out of error messages and logs,
  and a database failure is logged by error type only. `inputs` are stored because a disputed
  decision must be reproducible and monitoring (D-021) needs production distributions.
- **Result.** 181 tests pass (17 new), and ruff is clean (now also over `migrations/`).
  End-to-end on a real server:
  - start on an unmigrated database: refused with `SchemaNotCurrent`
  - `make migrate`: revision 0001 applied
  - `/ready` returned ready, then one approve (586) and one decline (500, 4 reasons), each with
    its own `decision_id`; a 422 request was not stored
  - 202 rows stored with their inputs; latency including the DB write about **16 ms/request**
- **Bug found and fixed.** The first migration doubled the constraint names
  (`ck_decisions_ck_decisions_pd`): Alembic applied the naming convention again on top of the full
  name. Alembic's autogenerate comparison ignores CHECK constraints, so a dedicated test now
  compares constraint and index names between the migration and the model. The migration was only
  ever applied to a throwaway database. *Rule from now on: a migration that has run anywhere real
  is never edited; a new one is added.*
- **Alternatives.**
  - *Return the decision and log the write failure* (fail open): issues unrecorded decisions.
  - *Retry in the background:* lost on a crash. The robust version is a transactional outbox or a
    message queue, which is more infrastructure than this demo needs.
  - *`create_all()` at start-up:* cannot evolve a schema, and gives no review step.
  - *Store only a hash of the inputs:* cannot reproduce a decision or feed monitoring.
  - *SQLAlchemy 2.1:* blocked by Smart App Control. 2.0.58 is pinned and falls back to pure Python.
- **Open risks.**
  - *Append-only is enforced by the code only.* In Postgres, a production setup would also revoke
    UPDATE/DELETE from the app's database role (planned for the Docker step).
  - *Retention:* Regulation B requires 25 months (12 CFR 1002.12). Nothing purges old rows yet.
  - *Encryption at rest and access control on `inputs`* are database and infrastructure duties.
  - *SQLite drops time zones:* timestamps are stored as UTC by convention.
  - *No idempotency key:* a client that retries after a timeout can create two records for one
    application. A client-supplied `application_id` with a unique constraint would fix this.

## D-026 Docker: one locked image, migrate once before the API, least-privilege database roles
*Approved, 2026-11-21. Locked image, migrate-once, app role SELECT+INSERT only.*

- **What.**
  - **`Dockerfile`.** Two stages: dependencies are installed into a virtualenv, then copied into a
    clean `python:3.12-slim` runtime. The image runs as non-root user `app` (uid 10001), and the
    app files are owned by root, so the process **cannot modify the model**. A `HEALTHCHECK` calls
    `/health`, and the container runs one uvicorn process.
  - **`.dockerignore` is an allow-list.** The image contains only `src/`, `configs/`,
    `migrations/`, `alembic.ini`, the two served model files and the test lock. No `data/`, `.env`,
    `.git`, tests, notebooks or challenger model.
  - **`docker-compose.yml`** runs `db` (Postgres 17), then a one-off `migrate` job, then `api`.
    The API starts only after `migrate` exits 0 (`service_completed_successfully`), and its port is
    bound to `127.0.0.1` only.
  - **`requirements/serve.lock`** pins all 41 packages, resolved inside `python:3.12-slim` itself
    (`make lock`).
  - **`.env.example`** is committed. The real `.env` (random passwords) is git-ignored.
  - **`.gitattributes`** forces LF line endings on `*.sh`.
  - **`scripts/smoke.py`** checks a running API end to end (standard library only).
- **Database roles (`docker/postgres/10-roles.sh`).**
  - `scorecard_owner` runs the migrations.
  - `scorecard_app` (the API) gets **SELECT and INSERT only**, by default privileges on every table
    the owner creates.
  - `PUBLIC` loses CONNECT on the database and all rights on schema `public`.
  - This makes the audit log **append-only in the database itself** and closes the D-025 open
    risk.
- **Result (run on the laptop, 2026-11-21).**
  - The image builds (906 MB), and the stack is up in about 26 s: db healthy, migrate exit 0 on
    Postgres (transactional DDL), api healthy.
  - `scripts/smoke.py` passes: health, approve 586, decline 500 with 4 reasons, age 17 gets a 422,
    and the ids differ.
  - As `scorecard_app`, UPDATE, DELETE, DROP TABLE and CREATE TABLE are all refused.
  - In the container: the user is uid 10001, and writing the model gets "Permission denied".
  - From the API container, a wrong password is refused. Inside the db container, the official
    image trusts local connections. That is the image's default and is not reachable from outside.
  - **Outage drill:** with Postgres stopped, `/health` returned 200, `/ready` 503, and `/score` 503
    with `Retry-After: 5` and no decision issued. The log says `OperationalError` only. After
    Postgres restarted, the API recovered **without a restart** (`pool_pre_ping`).
  - 182 tests pass and ruff is clean.
- **Fixed along the way.** `scoring.py` imported `sha256` from `pipelines/evaluate.py`, which
  pulled LightGBM into the API. LightGBM needs `libgomp`, which the slim image lacks, so **the
  container would have crashed at start-up**. `sha256` moved to `scorecard/integrity.py`, and a test
  (run in a fresh interpreter) fails if the API ever imports LightGBM or matplotlib again.
- **Why.**
  - *Migrate once:* three replicas migrating at the same moment race each other (failed
    `CREATE TABLE`, crash loops, half-applied changes on databases without transactional DDL).
  - *Two database roles:* the account that serves traffic cannot alter or delete history even if
    the API were compromised.
  - *Lock resolved on Linux:* the lock must match the platform that installs it. The Windows pins
    are constrained by Smart App Control and are not the same question.
  - *Allow-list `.dockerignore`:* a new secret or data file can never slip into an image by default.
- **Alternatives.**
  - *Migrations in the API's start command:* the replica race.
  - *A single `python:3.12` stage:* about twice the size, with compilers in production.
  - *Alpine base:* the musl C library breaks or slows numpy, scipy and scikit-learn wheels.
  - *`pip install .` in the image:* `config.py` locates `configs/` from the source tree, so
    `PYTHONPATH=/app/src` keeps the dev and container layouts identical.
  - *uvicorn `--workers N`:* hides scaling inside one container. One process per container lets
    the orchestrator scale and restart each instance.
  - *pip-tools/uv with hashes:* stronger supply-chain pinning (`--require-hashes`), worth adding in
    CI.
- **Open risks.**
  - *Image size.* 906 MB: training-only libraries (LightGBM, matplotlib) are installed but never
    imported. Splitting `train` and `serve` dependency groups would trim it.
  - *Postgres pinned by major version only* (`17-alpine`). Pin by digest for full reproducibility.
  - *No authentication or TLS on the API.* Localhost-only in compose; in a bank this sits behind an
    API gateway (mTLS or OAuth) and is never exposed directly.
  - *`make` is not installed on the dev laptop.* The Docker targets were run as their underlying
    commands, and the Makefile recipes run as written in CI (Linux).

## D-027 CI and pre-commit: one rule set everywhere, a green run must mean the tests ran, byte-exact artifacts in git
*Approved, 2026-11-21. One rule set locally and in CI; missing artifacts fail; artifacts stored byte-exact.*

- **What.**
  - **`.pre-commit-config.yaml`** has 12 hooks.
    - File hygiene: large files over 600 KB, private keys, merge markers, YAML/TOML syntax, final
      newline, trailing whitespace, LF line endings.
    - `ruff check --fix` and `ruff format`.
    - Two blockers: **no `data/` files** (Kaggle licence) and **no `.env`**.
    - `pytest` on `git push`.
  - **`.github/workflows/ci.yml`** runs on every push to `main` and every PR.
    - `checks` job: install, `pre-commit run --all-files`, then `pytest -rs` with
      `SCORECARD_REQUIRE_ARTIFACTS=1`.
    - `docker` job: throwaway random secrets, then `make up`, then `make smoke` (end to end, plus
      the API's DB role refused UPDATE), with logs on failure and always a tear-down.
    - Hardening: `permissions: contents: read`, actions **pinned by commit SHA**, timeouts, and
      cancellation of superseded runs.
  - **Dev tools pinned**: pytest 9.2.0, ruff 0.17.5, pre-commit 4.7.0, pre-commit-hooks 6.0.0.
  - **`make hooks`** installs the git hooks.
- **Decisions inside.**
  1. *A green run must mean the tests ran.* With `SCORECARD_REQUIRE_ARTIFACTS=1`, a missing model
     **fails** CI instead of skipping. The skew test needs the Kaggle splits, which are never in
     git, so it is the one reported skip ("splits not built"). It is a local pre-release check.
  2. *All hooks run from the project's pinned environment* (`language: system`,
     `python -m pre_commit_hooks.*`). Pre-commit's own downloaded environments were **blocked by
     Smart App Control** (`WinError 4551` on their `.exe` launchers). Calling them as modules goes
     through the trusted `python.exe`, and the versions are the same on the laptop and in CI.
  3. *Generated artifacts are never rewritten.* `models/` and `reports/` are excluded from every
     hook. The **only** files without a final newline were the model JSONs and the test lock, so
     an unconfigured end-of-file fixer would have changed `scorecard.json`'s SHA-256 on the first
     commit, and the API would have refused the model.
- **Bug found before the first push: line endings.**
  - The model JSONs and the test lock were written on Windows with CRLF, and their fingerprints
    are over those bytes.
  - With `core.autocrlf=true` and no attributes, git would have stored them with LF, so every
    clone (CI, Docker builds on other machines) would have failed the model check.
  - Fix in `.gitattributes`: `models/**` and `reports/**` are `-text` (stored byte-for-byte), and
    everything else is `text=auto eol=lf`.
  - 18 source files that my edit scripts had turned into CRLF (Python's `write_text` on Windows)
    were converted back to LF. The model and lock bytes were verified unchanged.
- **Also found.** A real `NameError` in `notebooks/01_data_audit.ipynb` (it imported `numpy as np`
  and then printed `numpy.__version__`); fixed. And `docker/postgres/10-roles.sh` must be committed
  **executable** (`git add --chmod=+x`). Otherwise Postgres's entrypoint *sources* it, and its
  `set -eu` would leak into the entrypoint shell.
- **Result.**
  - On the laptop, all 12 hooks pass and change nothing. The guards were checked: a data CSV and
    `.env` are refused; `models/scorecard.json` and the lock are skipped by the fixers (fingerprint
    still `27f4de0069b8`).
  - **CI simulated in clean Linux:** the files to be committed were committed with
    `core.autocrlf=true` and Windows file modes, then cloned fresh.
    - The model and reason-reference bytes **match the test lock**.
    - `pre-commit run --all-files` exits 0.
    - pytest: **181 passed, 1 skipped** (skew test, by design).
    - `make -n up smoke lock` parses.
  - The `docker` job's commands were verified in D-026.
- **Alternatives.**
  - *Standard remote hook repos:* pre-commit's usual model, blocked on this laptop, and a second
    copy of every tool version.
  - *Lint in CI only:* problems found minutes later and after they are already in history.
  - *Tests in the pre-commit stage:* about 12 s on every commit discourages small commits, so
    tests run on push.
  - *Commit the data so the skew test runs in CI:* breaks the licence.
  - *Actions pinned by tag:* a moved or hijacked tag would run new code in our pipeline.
- **Open risks.**
  - *Artifact hashes include the platform's line endings.* A rebuild on Linux would write LF JSON,
    a different SHA-256 from the CRLF lock. Writing with `newline="\n"`, or hashing canonical JSON,
    fixes this but needs a **re-lock**, which is a governance event. It is deliberately not done
    silently here.
  - *No branch protection yet.* "CI must pass before merge" is a GitHub repository setting for the
    owner to switch on.
  - *The skew test never runs in CI.* Run `make check` locally (data present) before any release.

---

## D-028 Documentation: architecture, assumptions register, business case
*Approved, 2026-11-21. Figures copied from generated reports; update the documents on any rebuild.*

- **What.** Three documents in `docs/`, written for reviewers who have not read the code:
  - [architecture.md](architecture.md): the offline pipeline, the scoring service, the audit table,
    deployment and quality gates, with diagrams;
  - [assumptions.md](assumptions.md): every number not learned from data, labelled Benchmark /
    Assumption / Illustrative / Data fact, plus limitations and the open-risk register (R-1 to R-10);
  - [business_case.md](business_case.md): test-set results, policy outcome at cutoff 549, the
    Illustrative value (+$1.10M per 10,000 applicants vs approve-all), sensitivity, the scorecard
    vs challenger trade-off, KPIs and next steps.
- **Why.** A model risk reviewer or credit committee needs the assumptions and the money case in
  one place, separate from code. Every figure is copied from a generated report, so the documents
  can be checked against `reports/`.
- **Alternatives.** Leaving the information spread across the decision log and reports (complete,
  but hard to review); a single long README (mixes setup instructions with governance content).
- **Open risks.** The documents are written by hand. If the model is rebuilt with different
  results, they must be updated too. Generating them from the reports would remove this risk.

---

## D-029 Model card with subgroup performance; README and data guide
*Approved, 2026-11-21. Subgroups on valid; 18-29 approval ratio disclosed as R-11.*

- **What.** [reports/model_card.md](../reports/model_card.md) follows Mitchell et al. (2019) and
  adds validator items: intended and out-of-scope use, data, features, test performance,
  calibration, subgroups, fairness, limitations, monitoring. A new module,
  `python -m scorecard.evaluation.subgroups`, writes `reports/subgroup_performance.md` and is
  part of the training pipeline. It reports, for each group:
  - size, bad rate and mean PD;
  - AUC (left blank under 30 bads);
  - approval rate and goods declined at cutoff 549;
  - approval ratio to the best group (four-fifths rule, **Benchmark**).

  Groups are age band (split at 62, the Regulation B threshold) and income reported / not.
  `README.md` and `data/README.md` are rewritten.
- **Data used.** **Valid**, not test: test was scored once (D-020), and a new analysis on it would be
  a second look. The subgroup numbers do not change any model choice.
- **Findings.**
  - Ages 18-29: approval ratio **0.79** (under 0.8), AUC 0.795. The gap mostly follows observed
    risk (bad rate 11.2% vs 2.9% for 62+). The PD over-prediction (12.18% vs 11.16%) is about 1.2
    standard errors, within noise. Added as risk R-11.
  - Income not reported: PD 4.88% vs bad rate 5.50% (about -1.9 standard errors, borderline
    under-prediction). This supports the income-verification policy note (R-3).
  - Ages 62+: highest approval (96.0%), so they are not disadvantaged.
- **Why.** A model can perform well overall and badly for one group. Validators and fair-lending
  reviews ask for exactly these tables. Generating them in code keeps them reproducible.
- **Alternatives.**
  - Subgroups on test: more independent, but reopens the locked set.
  - Fairness metrics on protected classes: race, sex and marital status are not in the data. A
    lender would use proxy methods (BISG).
  - Removing `age` now: a policy choice. Its approval impact should be measured first, so it is
    left to the owner.
- **Open risks.** The model card is hand-written (see D-028). Subgroups by protected class are untested.

---

## D-030 Underwriter what-if app: Streamlit on the scoring core, simulation only
*Proposed, 2026-11-21.*

- **What.** `app/streamlit_app.py` (`make app`, http://127.0.0.1:8501). Credit staff enter an
  application and see:
  - the score, PD and outcome;
  - the decline reasons as they would appear on the notice;
  - where each point comes from.

  They then pick fields to change and see the new score, PD and outcome side by side, with the
  per-feature point changes. A new core method, `ScoringService.breakdown()`, returns the value,
  bin, points, typical points and difference per scored feature. Its points sum **exactly** to the
  score.
- **Design rules.**
  1. **Same core as the API.** The app calls `ScoringService`, so its numbers cannot drift from
     live decisions. The model is loaded once (`st.cache_resource`) and verified against the
     test lock.
  2. **Simulation only.** Nothing is written to the audit log, and a banner says so. Decisions
     have one channel (`POST /score`), so every real decision has exactly one record. A
     what-if must never be mistaken for one.
  3. **Input bounds mirror the API's rules** (age 18+, non-negative counts, "not reported"
     only for income and dependents). The core still validates.
- **Dependencies.** A separate `app` extra (`streamlit==1.67.0`, `pyarrow==23.0.1`), so the API
  image does not include Streamlit. pyarrow is pinned to 23 because later releases require
  NumPy 2, and core pins NumPy 1.26 (Smart App Control). CI installs the extra, so the app tests
  run rather than skip.
- **Result.**
  - 8 new tests: 4 for `breakdown`, and 4 that run the real app headless with Streamlit's
    `AppTest` against the lock-verified model.
  - The app reproduces the README decline (487, reasons DLQ_90, UTIL_HIGH, DLQ_30, AGE).
  - The utilisation what-if (0.95 -> 0.25) gives 519, +32 points, still a decline.
  - Full suite 195 passed.
- **Why.** Additive points are the scorecard's main advantage over the challenger, and a
  breakdown that sums exactly to the score is how that advantage reaches an underwriter. What-if
  answers the questions an applicant actually asks ("what would get me approved?") from the
  model itself.
- **Alternatives.**
  - Calling the API over HTTP: needs the server and database running, and would write audit
    records for simulations.
  - A React front end: more work, with no modelling benefit for a demo.
  - SHAP-style explanations: unnecessary when the model is additive.
- **Open risks.**
  - No login. Run it locally only; a deployed version would need authentication, since it
    reveals the scorecard's points.
  - "What would get me approved" is not counter-offer advice: changes must be real changes in
    the applicant's situation, not edits to the form.
