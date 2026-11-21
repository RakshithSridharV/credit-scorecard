"""What changes if `age` is removed from the scorecard? (risk R-11, D-032)

The deployed model gives applicants aged 18-29 an approval ratio of 0.79 against
the best age band, just under the four-fifths screen. Removing age is the obvious
remedy, but the other features may carry the same information (a young applicant
also tends to have higher utilisation and recent late payments): the proxy
problem. This experiment measures both sides of the trade-off:

* what it costs: ranking (AUC), calibration and Illustrative profit;
* what it fixes: approval rates and ratios by age band.

The challenger without age is built exactly like the deployed model (same
binning, same unpenalised fit, same out-of-fold master scale, same break-even
cutoff rule), so the only difference is the missing feature.

    python -m scorecard.evaluation.age_ablation

Train fits, VALID compares. Test is not reopened (D-020). Nothing deployed changes:
this is evidence for the owner's decision, not a new model.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.cutoff import (
    Economics,
    economics_from_config,
    policy_cutoff,
    row_at,
    strategy_table,
)
from scorecard.evaluation.metrics import summary
from scorecard.evaluation.subgroups import AGE_LABELS, FOUR_FIFTHS, age_band, subgroup_table
from scorecard.models.challenger import paired_bootstrap_auc_gap
from scorecard.models.scorecard import Scaling, Scorecard, oof_scores

AGE = "age"
YOUNG, OLD = AGE_LABELS[0], AGE_LABELS[-1]  # "18-29" and "62+" (Regulation B group)
PCT = ["approval_rate", "bad_rate_approved"]


def fit_without(
    train: pd.DataFrame,
    features: list[str],
    drop: str,
    binning: dict[str, Any],
    scaling: Scaling,
    pd_floor: float,
    seed: int,
) -> Scorecard:
    """The deployed recipe (D-016) on `features` minus `drop`."""
    cols = [f for f in features if f != drop]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, train[TARGET]))
    sc = Scorecard(cols, binning, scaling).fit(train, train[TARGET])
    return sc.calibrate(oof_scores(train, cols, binning, scaling, folds), train[TARGET], pd_floor)


def model_row(sc: Scorecard, data: pd.DataFrame, cutoff: int, econ: Economics) -> dict[str, float]:
    """Headline numbers for one model on one data set, at its own policy cutoff."""
    y, scores, pds = data[TARGET], sc.score(data), sc.score_pd(data)
    m = summary(y, pds)
    policy = row_at(strategy_table(scores, y, econ), cutoff)
    young = (age_band(data[AGE]) == YOUNG).to_numpy()
    return {
        "features": len(sc.features),
        "cutoff": cutoff,
        "auc": m["auc"],
        "gini": m["gini"],
        "ks": m["ks"],
        "mean_pd / bad_rate": m["mean_pd"] / m["bad_rate"],
        "approval_rate": policy["approval_rate"],
        "bad_rate_approved": policy["bad_rate_approved"],
        "profit_per_10k": policy["profit_per_10k"],
        # Proxy strength: how well the score alone tells an 18-29 applicant from
        # an older one. 0.5 = the score carries no age information.
        "score_auc_for_18_29": roc_auc_score(young, -scores),
    }


def compare_by_age(
    with_age: pd.DataFrame, without_age: pd.DataFrame, cols: tuple[str, ...]
) -> pd.DataFrame:
    """Side-by-side subgroup columns: deployed vs without age, plus the change."""
    out = {}
    for c in cols:
        out[f"{c} (deployed)"] = with_age[c]
        out[f"{c} (no age)"] = without_age[c]
        out[f"{c} change"] = without_age[c] - with_age[c]
    return pd.DataFrame(out)


def paired_bootstrap_age(
    bands: pd.Series,
    y: ArrayLike,
    approved_dep: ArrayLike,
    approved_no: ArrayLike,
    pd_dep: ArrayLike,
    pd_no: ArrayLike,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """95% intervals for the age effects, resampling the SAME rows for both models.

    Pairing (as in D-017) matters: both models score the same applicants, so
    their group rates move together from sample to sample, and the change
    between them is far more precise than either rate on its own.
    """
    codes = pd.Categorical(bands, categories=AGE_LABELS).codes
    if (codes < 0).any():
        raise ValueError("every row needs an age band")
    k, young, old = len(AGE_LABELS), 0, len(AGE_LABELS) - 1
    y_, a_dep, a_no = (np.asarray(v, dtype=float) for v in (y, approved_dep, approved_no))
    p_dep, p_no = np.asarray(pd_dep, dtype=float), np.asarray(pd_no, dtype=float)

    def stats(idx: np.ndarray) -> list[float]:
        c = codes[idx]
        size = np.bincount(c, minlength=k)
        rate_dep = np.bincount(c, weights=a_dep[idx], minlength=k) / size
        rate_no = np.bincount(c, weights=a_no[idx], minlength=k) / size
        bad_old = np.bincount(c, weights=y_[idx], minlength=k)[old]
        ratio_dep = rate_dep[young] / rate_dep.max()
        ratio_no = rate_no[young] / rate_no.max()
        return [
            ratio_dep,
            ratio_no,
            ratio_no - ratio_dep,
            rate_no[old] - rate_dep[old],
            np.bincount(c, weights=p_dep[idx], minlength=k)[old] / bad_old,
            np.bincount(c, weights=p_no[idx], minlength=k)[old] / bad_old,
        ]

    names = [
        f"{YOUNG} approval ratio, deployed",
        f"{YOUNG} approval ratio, without age",
        f"{YOUNG} approval ratio, change",
        f"{OLD} approval rate, change",
        f"{OLD} mean PD / bad rate, deployed",
        f"{OLD} mean PD / bad rate, without age",
    ]
    rng = np.random.default_rng(seed)
    # A resample with no one in a band, or no bads among 62+ (only possible on tiny
    # data), gives NaN or inf for that draw; it is skipped rather than allowed to
    # spoil the interval.
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.array([stats(rng.integers(0, len(y_), len(y_))) for _ in range(n)])
        estimate = stats(np.arange(len(y_)))
    draws[~np.isfinite(draws)] = np.nan
    return pd.DataFrame(
        {
            "estimate": estimate,
            "ci_low": np.nanpercentile(draws, 2.5, axis=0),
            "ci_high": np.nanpercentile(draws, 97.5, axis=0),
        },
        index=pd.Index(names, name="quantity"),
    )


def main() -> None:
    cfg = load_config()
    interim = resolve_path(cfg["data"]["interim_dir"])
    train, valid = load_split("train", interim), load_split("valid", interim)
    selected = json.loads(
        resolve_path(cfg["artifacts"]["selected_features"]).read_text(encoding="utf-8")
    )["features"]
    sc_cfg = dict(cfg["scorecard"])
    pd_floor = sc_cfg.pop("pd_floor")
    econ = economics_from_config(cfg)

    deployed = Scorecard.from_json(resolve_path(cfg["artifacts"]["scorecard"]))
    no_age = fit_without(
        train, selected, AGE, cfg["binning"], Scaling(**sc_cfg), pd_floor, cfg["split"]["seed"]
    )
    cut_dep, cut_no = policy_cutoff(deployed, econ), policy_cutoff(no_age, econ)

    models = pd.DataFrame(
        {
            "deployed (with age)": model_row(deployed, valid, cut_dep, econ),
            "without age": model_row(no_age, valid, cut_no, econ),
        }
    ).T
    models["profit_per_10k_change"] = (
        models["profit_per_10k"] - models.loc["deployed (with age)", "profit_per_10k"]
    )
    models["features"] = models["features"].astype(int)
    models["cutoff"] = models["cutoff"].astype(int)

    y = valid[TARGET]
    bands = age_band(valid[AGE])
    sub_dep = subgroup_table(bands, y, deployed.score_pd(valid), deployed.score(valid), cut_dep)
    sub_no = subgroup_table(bands, y, no_age.score_pd(valid), no_age.score(valid), cut_no)
    by_age = compare_by_age(sub_dep, sub_no, ("approval_rate", "approval_ratio"))
    calib = compare_by_age(sub_dep, sub_no, ("pd_to_bad_rate",))
    coefs = no_age.coef_table_[["coef", "z"]]
    boot = paired_bootstrap_age(
        bands,
        y,
        deployed.score(valid) >= cut_dep,  # same approval rule as the API
        no_age.score(valid) >= cut_no,
        deployed.score_pd(valid),
        no_age.score_pd(valid),
        cfg["fairness"]["n_bootstrap"],
        cfg["split"]["seed"],
    )

    # AUC(without age) - AUC(deployed): negative = ranking lost by dropping age.
    auc_gap = paired_bootstrap_auc_gap(
        y,
        deployed.score_pd(valid),
        no_age.score_pd(valid),
        cfg["fairness"]["n_bootstrap"],
        cfg["split"]["seed"],
    )

    young = boot.loc[f"{YOUNG} approval ratio, without age"]
    verdict = "clears" if young["estimate"] >= FOUR_FIFTHS else "stays below"
    old_cal = boot.loc[f"{OLD} mean PD / bad rate, without age"]

    nl = "\n"
    sections = [
        "# Experiment: the scorecard without `age` (risk R-11)" + nl,
        "Both models fitted on **train** with the same recipe (D-016); compared on **valid** "
        f"({len(valid):,} rows), each at its own break-even cutoff (D-018). Test is not reopened "
        "(D-020). Profit is **Illustrative**. Generated by "
        "`python -m scorecard.evaluation.age_ablation`." + nl,
        f"**Result.** Without age, the {YOUNG} approval ratio is **{young['estimate']:.3f}** "
        f"(95% CI {young['ci_low']:.3f} to {young['ci_high']:.3f}), which {verdict} the "
        f"four-fifths screen ({FOUR_FIFTHS}, **Benchmark**) on the point estimate. The cost "
        f"falls on {OLD}: mean PD / bad rate becomes **{old_cal['estimate']:.2f}** "
        f"(95% CI {old_cal['ci_low']:.2f} to {old_cal['ci_high']:.2f}), so their risk is "
        "overstated." + nl,
        "## Cost: the whole portfolio" + nl,
        "`score_auc_for_18_29` = how well the score alone separates 18-29 applicants from older "
        "ones (0.5 = no age information). It stays well above 0.5 without age when the other "
        "features act as proxies." + nl,
        md_table(models.round(4), pct_cols=PCT),
        nl + f"AUC change without age: **{auc_gap['gap']:+.4f}** (paired bootstrap 95% CI "
        f"{auc_gap['ci_low']:+.4f} to {auc_gap['ci_high']:+.4f})." + nl,
        nl + "## Effect: approval by age band" + nl,
        md_table(by_age.round(4), pct_cols=[c for c in by_age.columns if "rate" in c]),
        nl + "## Calibration by age band (mean PD / bad rate)" + nl,
        "Without age, the model can no longer tell a young applicant from an older one with the "
        "same file. The age effect is spread over everyone: PD rises for the bands age had "
        "rewarded and falls for those it had penalised." + nl,
        md_table(calib.round(4)),
        nl + "## Uncertainty: paired bootstrap on valid" + nl,
        f"{cfg['fairness']['n_bootstrap']} resamples of valid rows (seed {cfg['split']['seed']}), "
        "the same rows for both models, so the changes are measured precisely." + nl,
        md_table(boot.round(4)),
        nl + "## Coefficients without age" + nl,
        "All must stay negative (WoE positive = safer)." + nl,
        md_table(coefs.round(4)),
    ]
    report = resolve_path(cfg["reports"]["age_ablation"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8", newline="\n")

    pd.set_option("display.width", 220)
    print(models.round(4).to_string(), nl)
    print(by_age.round(4).to_string(), nl)
    print(boot.round(4).to_string(), nl)
    print("AUC change:", {k: round(v, 4) for k, v in auc_gap.items()}, nl)
    print(calib.round(4).to_string(), nl)
    print(coefs.round(4).to_string(), nl)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
