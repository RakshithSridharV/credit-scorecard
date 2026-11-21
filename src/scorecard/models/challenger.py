"""Challenger model: LightGBM, used to measure what the scorecard gives up (D-017).

The scorecard is additive: one points value per bin, no interactions. Gradient
boosting can learn interactions and finer shapes. The AUC gap between the two
is the price of interpretability. It is a benchmark, not a candidate for
deployment.

Three variants separate where any gap comes from:
  * LightGBM on the scorecard's 7 features -> gain from model flexibility alone
  * LightGBM on all 10 features             -> plus the 3 features selection dropped
  * LightGBM on all 10, monotone            -> what is left once every feature must
                                               move risk in the direction credit
                                               logic expects (what a bank could defend)

    python -m scorecard.models.challenger

tunes each variant by 5-fold CV on train (same folds as the scorecard), compares
on valid with a paired bootstrap, saves the monotone model and writes
reports/challenger.md. No test data.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import FEATURES, TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.calibration import calibration_in_the_large, calibration_table
from scorecard.evaluation.metrics import summary
from scorecard.models.scorecard import Scorecard

Folds = list[tuple[np.ndarray, np.ndarray]]

# Expected effect of a HIGHER value on risk: +1 riskier, -1 safer, 0 no constraint.
# Set from credit logic, not read off the data, so the constraint is a prior the
# data must live with. The WoE trends (reports/woe_summary.md) agree.
MONOTONE_DIRECTION: dict[str, int] = {
    "RevolvingUtilizationOfUnsecuredLines": +1,
    "age": -1,
    "NumberOfTime30-59DaysPastDueNotWorse": +1,
    "DebtRatio": +1,
    "MonthlyIncome": -1,
    # Few lines = thin file (risky), very many = credit hungry (risky): U-shape, leave free.
    "NumberOfOpenCreditLinesAndLoans": 0,
    "NumberOfTimes90DaysLate": +1,
    # No mortgage is riskier than one, but many property loans can be leverage: leave free.
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": +1,
    "NumberOfDependents": +1,
}


def base_params(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fixed LightGBM settings (everything not tuned)."""
    return {
        "objective": "binary",  # log loss: the output is a probability, not just a rank
        "metric": "auc",
        "learning_rate": cfg["learning_rate"],
        "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"],
        "bagging_freq": 1,
        "lambda_l2": cfg["lambda_l2"],
        "seed": seed,
        "deterministic": True,  # same result every run on the same machine
        "force_row_wise": True,  # avoids a run-time choice that can change results
        "verbosity": -1,
    }


def monotone_vector(features: Sequence[str]) -> list[int]:
    """LightGBM wants one direction per column, in column order."""
    return [MONOTONE_DIRECTION[f] for f in features]


@dataclass(frozen=True)
class Tuned:
    params: dict[str, Any]
    n_rounds: int
    cv_auc: float
    cv_sd: float
    grid: pd.DataFrame  # every grid point tried, for the report


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    folds: Folds,
    params: dict[str, Any],
    grid: dict[str, list[Any]],
    max_rounds: int,
    early_stopping: int,
) -> Tuned:
    """Grid search with CV early stopping; picks the highest mean CV AUC.

    Early stopping also chooses the number of trees, on the same folds. That
    makes the best CV AUC slightly optimistic, which is why valid, not CV, is
    the number we compare.
    """
    rows, combos = [], []
    for values in itertools.product(*grid.values()):
        combo = dict(zip(grid.keys(), values, strict=True))
        combos.append(combo)
        p = {**params, **combo}
        res = lgb.cv(
            p,
            lgb.Dataset(X, y),  # fresh Dataset: bin settings may not be shared across params
            num_boost_round=max_rounds,
            folds=folds,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        mean, sd = res["valid auc-mean"], res["valid auc-stdv"]
        best = int(np.argmax(mean))
        rows.append({**combo, "n_rounds": best + 1, "cv_auc": mean[best], "cv_sd": sd[best]})
    table = pd.DataFrame(rows)
    i = int(table["cv_auc"].idxmax())
    # Take the winner from `combos`, not from the table row: a mixed-type row
    # comes back as floats, and LightGBM rejects num_leaves=15.0.
    top = table.iloc[i]
    return Tuned(
        {**params, **combos[i]},
        int(top["n_rounds"]),
        float(top["cv_auc"]),
        float(top["cv_sd"]),
        table,
    )


def fit_final(X: pd.DataFrame, y: pd.Series, tuned: Tuned) -> lgb.Booster:
    """Refit on all of train with the tuned settings and number of trees."""
    return lgb.train(tuned.params, lgb.Dataset(X, y), num_boost_round=tuned.n_rounds)


def paired_bootstrap_auc_gap(
    y: ArrayLike, p_base: ArrayLike, p_other: ArrayLike, n: int, seed: int
) -> dict[str, float]:
    """AUC(other) - AUC(base), resampling the SAME rows for both models.

    Pairing matters: both models score the same applicants, so their AUCs move
    together from sample to sample. Resampling them separately would overstate
    the uncertainty of the gap.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y, p_base, p_other))
    rng = np.random.default_rng(seed)
    gaps = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        gaps[i] = roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx])
    return {
        "gap": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
        "ci_low": float(np.percentile(gaps, 2.5)),
        "ci_high": float(np.percentile(gaps, 97.5)),
        "share_gap_positive": float((gaps > 0).mean()),
    }


def gain_importance(model: lgb.Booster) -> pd.Series:
    """Share of total split gain per feature: how much each feature drives the model."""
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False).rename("gain_share")


def monotone_violations(
    model: lgb.Booster, X: pd.DataFrame, feature: str, direction: int, n_rows: int = 200
) -> dict[str, float]:
    """How often, and how far, PD moves the wrong way as `feature` sweeps its range.

    Each sampled row is copied across a grid of values for one feature, all else
    fixed (an individual conditional expectation curve). `rows` counts rows with
    any wrong-way step; `max_pp` is the largest such step in PD percentage
    points, so a count driven by tiny wiggles is not mistaken for a real reversal.
    """
    rows = X.sample(n=min(n_rows, len(X)), random_state=0)
    grid = np.unique(np.nanquantile(X[feature], np.linspace(0, 1, 25)))
    # One frame of len(rows) * len(grid) rows, predicted in a single call.
    sweep = rows.loc[rows.index.repeat(len(grid))].reset_index(drop=True)
    sweep[feature] = np.tile(grid, len(rows))
    p = model.predict(sweep[model.feature_name()]).reshape(len(rows), len(grid))
    wrong = np.clip(-direction * np.diff(p, axis=1), 0, None)  # size of each wrong-way step
    return {"rows": int((wrong > 1e-12).any(axis=1).sum()), "max_pp": float(wrong.max() * 100)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    ch, seed = cfg["challenger"], cfg["split"]["seed"]
    interim = resolve_path(cfg["data"]["interim_dir"])
    train, valid = load_split("train", interim), load_split("valid", interim)
    selected = json.loads(
        resolve_path(cfg["artifacts"]["selected_features"]).read_text(encoding="utf-8")
    )["features"]
    # Same folds as the scorecard, so CV numbers are comparable fold for fold.
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, train[TARGET]))

    sc = Scorecard.from_json(resolve_path(cfg["artifacts"]["scorecard"]))
    y_va = valid[TARGET]
    p_sc = sc.score_pd(valid)  # the deployed output: integer score -> master scale PD

    params = base_params(ch, seed)
    variants: dict[str, tuple[list[str], dict[str, Any]]] = {
        "LightGBM, 7 scorecard features": (selected, params),
        "LightGBM, all 10": (FEATURES, params),
        "LightGBM, all 10, monotone": (
            FEATURES,
            {**params, "monotone_constraints": monotone_vector(FEATURES)},
        ),
    }

    rows, models, grids, preds = [], {}, {}, {}
    sc_metrics = summary(y_va, p_sc)
    rows.append(
        {
            "model": "Scorecard (D-016, deployed PD)",
            "cv_auc": np.nan,
            "n_rounds": np.nan,
            "num_leaves": np.nan,
            "min_child_samples": np.nan,
            **{f"valid_{k}": v for k, v in sc_metrics.items() if k != "bad_rate"},
        }
    )
    for name, (cols, p) in variants.items():
        tuned = tune(
            train[cols], train[TARGET], folds, p, ch["grid"], ch["max_rounds"], ch["early_stopping"]
        )
        model = fit_final(train[cols], train[TARGET], tuned)
        pred = model.predict(valid[cols])
        models[name], grids[name], preds[name] = model, tuned.grid, pred
        m = summary(y_va, pred)
        rows.append(
            {
                "model": name,
                "cv_auc": tuned.cv_auc,
                "n_rounds": tuned.n_rounds,
                "num_leaves": tuned.params["num_leaves"],
                "min_child_samples": tuned.params["min_child_samples"],
                **{f"valid_{k}": v for k, v in m.items() if k != "bad_rate"},
            }
        )
    results = pd.DataFrame(rows).set_index("model")

    gaps = pd.DataFrame(
        {
            name: paired_bootstrap_auc_gap(y_va, p_sc, pred, ch["n_bootstrap"], seed)
            for name, pred in preds.items()
        }
    ).T.rename_axis("challenger vs scorecard")

    mono_name = "LightGBM, all 10, monotone"
    importance = pd.DataFrame({n: gain_importance(models[n]) for n in variants}).fillna(0)
    importance = importance.sort_values("LightGBM, all 10", ascending=False)
    violations = pd.concat(
        {
            n: pd.DataFrame(
                {
                    f: monotone_violations(models[n], valid[FEATURES], f, d)
                    for f, d in MONOTONE_DIRECTION.items()
                    if d != 0
                }
            ).T
            for n in ("LightGBM, all 10", mono_name)
        },
        axis=1,
    )
    violations.columns = [f"{m} | {c}" for m, c in violations.columns]
    violations = violations.rename_axis("feature")

    pct = ["predicted_pd", "observed_bad_rate"]
    calib_ch = calibration_table(y_va, preds[mono_name])
    calib_ch["ratio_mean_pd"] = calibration_in_the_large(y_va, preds[mono_name])

    models[mono_name].save_model(str(resolve_path(cfg["artifacts"]["challenger"])))

    nl = "\n"
    sections = [
        "# Challenger model (LightGBM)" + nl,
        "Tuned by 5-fold CV on **train** (the scorecard's folds); compared on **valid**. "
        "Test is not used. Generated by `python -m scorecard.models.challenger`." + nl,
        "## Results (D-017)" + nl,
        md_table(results, pct_cols=["valid_mean_pd"]),
        f"{nl}Valid bad rate: {y_va.mean():.2%}.{nl}",
        "## AUC gap over the scorecard, paired bootstrap on valid" + nl,
        f"{ch['n_bootstrap']} resamples of valid rows (seed {seed}); "
        "95% interval from the 2.5th and 97.5th percentiles." + nl,
        md_table(gaps),
        nl + "## Where the challenger's signal comes from (share of split gain)" + nl,
        md_table(importance),
        nl + "## Monotonicity check" + nl,
        "For 200 valid rows, sweep one feature over 25 quantiles with everything else fixed; "
        "`rows` = rows whose PD moves against credit logic at any step; "
        "`max_pp` = largest such step, in PD percentage points." + nl,
        md_table(violations),
        nl + f"## Calibration of '{mono_name}' by PD decile (no recalibration)" + nl,
        md_table(calib_ch, pct_cols=pct),
        nl + "## Tuning grids (CV AUC on train)" + nl,
        *[f"### {n}{nl}{nl}{md_table(g, index=False)}{nl}" for n, g in grids.items()],
    ]
    report = resolve_path(cfg["reports"]["challenger"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8")

    pd.set_option("display.width", 220)
    print(results.round(4).to_string(), nl)
    print(gaps.round(4).to_string(), nl)
    print(importance.round(3).to_string(), nl)
    print(violations.to_string(), nl)
    print(calib_ch.round(4).to_string())
    print(f"{nl}Saved {cfg['artifacts']['challenger']}; report in {report}")


if __name__ == "__main__":
    main()
