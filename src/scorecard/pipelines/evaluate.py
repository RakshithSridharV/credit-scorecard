"""Final, one-time evaluation on the untouched TEST set (D-020).

Every choice so far used train (fitting) or valid (choosing), so valid numbers
are slightly optimistic. Test has never been scored. It is scored ONCE here,
with every artifact frozen, and that is the honest estimate for new applicants.

To make "once" checkable, the first run writes a lock file with the SHA-256 of
every frozen input. A second run is refused unless --force is given, and a
forced run is appended to the lock's history, so any re-look is on record.

    python -m scorecard.pipelines.evaluate            # first and only run
    python -m scorecard.pipelines.evaluate --force    # re-run, recorded in the lock
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score, roc_curve

from scorecard.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.calibration import calibration_in_the_large, calibration_table
from scorecard.evaluation.cutoff import (
    PER_N,
    economics_from_config,
    policy_cutoff,
    row_at,
    strategy_table,
)
from scorecard.evaluation.metrics import summary
from scorecard.explain.reason_codes import load_reference, reason_frequency, top_reasons
from scorecard.integrity import sha256  # re-exported: pipelines.train imports it here
from scorecard.models.challenger import paired_bootstrap_auc_gap
from scorecard.models.scorecard import Scorecard

SPECIAL_CODE_FLAG = "NumberOfTimes90DaysLate"  # NaN only where raw had 96/98 (D-006)


# --------------------------------------------------------------------------- #
# The once-only guard
# --------------------------------------------------------------------------- #
def fingerprints(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(p) for name, p in paths.items()}


def check_lock(lock: Path, force: bool) -> None:
    """Refuse a second look at test unless forced. Called BEFORE scoring."""
    if lock.exists() and not force:
        first = json.loads(lock.read_text(encoding="utf-8"))["runs"][0]
        raise RuntimeError(
            f"Test was already evaluated at {first['at']} (see {lock}). Looking again "
            "turns test into a second validation set. Re-run with --force to override; "
            "the re-run will be recorded."
        )


def record_run(lock: Path, prints: dict[str, str]) -> dict[str, Any]:
    """Append this run to the lock. Called only AFTER the report is written, so a
    crash part-way (no results seen) does not count as a look at test."""
    run: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprints": prints,
    }
    if lock.exists():
        state = json.loads(lock.read_text(encoding="utf-8"))
        run["forced"] = True
        run["same_artifacts_as_first"] = state["runs"][0]["fingerprints"] == prints
        state["runs"].append(run)
    else:
        state = {"runs": [run]}
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


# --------------------------------------------------------------------------- #
# Measures
# --------------------------------------------------------------------------- #
def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def bootstrap_ci(y: ArrayLike, p: ArrayLike, n: int, seed: int) -> pd.DataFrame:
    """95% percentile intervals for AUC, Gini and KS by resampling rows."""
    yy, pp = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    auc, ks = np.empty(n), np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(yy), len(yy))
        auc[i] = roc_auc_score(yy[idx], pp[idx])
        ks[i] = ks_stat(yy[idx], pp[idx])
    draws = {"auc": auc, "gini": 2 * auc - 1, "ks": ks}
    return pd.DataFrame(
        {
            k: {"ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5)}
            for k, v in draws.items()
        }
    ).T


def special_code_rows(df: pd.DataFrame) -> pd.Series:
    """Rows whose past-due fields were 96/98 in the raw data (risk R-1)."""
    return df[SPECIAL_CODE_FLAG].isna()


def score_decile_calibration(y: ArrayLike, p: ArrayLike, scores: np.ndarray) -> pd.DataFrame:
    t = calibration_table(y, p, rank_by=-scores)
    decile = pd.qcut(pd.Series(-scores).rank(method="first"), 10, labels=False).to_numpy()
    g = pd.Series(scores).groupby(decile)
    t["score_min"], t["score_max"] = g.min().to_numpy(), g.max().to_numpy()
    return t


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-run; recorded in the lock")
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help="run everything on VALID in place of test: no lock, no report (catches bugs "
        "without spending the one look at test)",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    ev, seed = cfg["evaluation"], cfg["split"]["seed"]
    art, interim = cfg["artifacts"], resolve_path(cfg["data"]["interim_dir"])
    eval_split = "valid" if args.rehearse else "test"
    frozen = {
        "scorecard": resolve_path(art["scorecard"]),
        "challenger": resolve_path(art["challenger"]),
        "reason_reference": resolve_path(art["reason_reference"]),
        "config": DEFAULT_CONFIG_PATH,
        "test_data": interim / f"{eval_split}.csv",
    }
    lock_path = resolve_path(ev["lock"])
    if not args.rehearse:
        check_lock(lock_path, args.force)
    prints = fingerprints(frozen)
    previous = (
        json.loads(lock_path.read_text(encoding="utf-8"))["runs"] if lock_path.exists() else []
    )
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.rehearse:
        runs_note = "REHEARSAL on valid: not a test result"
    elif not previous:
        runs_note = "first and only run"
    else:
        runs_note = f"run {len(previous) + 1}: FORCED re-run (first run at {previous[0]['at']})"

    sc = Scorecard.from_json(frozen["scorecard"])
    challenger = lgb.Booster(model_file=str(frozen["challenger"]))
    reference = load_reference(frozen["reason_reference"])
    econ = economics_from_config(cfg)
    cut = policy_cutoff(sc, econ)
    rc = cfg["reasons"]

    valid, test = load_split("valid", interim), load_split(eval_split, interim)

    # ---- headline: the same measures on valid (seen) and test (unseen)
    def headline(df: pd.DataFrame) -> dict[str, float]:
        m = summary(df[TARGET], sc.score_pd(df))
        m["mean_pd / bad_rate"] = calibration_in_the_large(df[TARGET], sc.score_pd(df))
        return m

    y_te, s_te, pd_te = test[TARGET].to_numpy(), sc.score(test), sc.score_pd(test)
    head = pd.DataFrame({"valid": headline(valid), "test": headline(test)})
    head.loc["rows"] = [len(valid), len(test)]
    ci = bootstrap_ci(y_te, pd_te, ev["n_bootstrap"], seed)

    calib = score_decile_calibration(y_te, pd_te, s_te)

    # ---- the business decision at the frozen cutoff
    strat = strategy_table(s_te, y_te, econ)
    at_cut, approve_all = row_at(strat, cut), strat.iloc[0]
    strat_va = strategy_table(sc.score(valid), valid[TARGET].to_numpy(), econ)
    policy = pd.DataFrame({"valid": row_at(strat_va, cut), "test": at_cut}).T
    value = at_cut["profit_per_10k"] - approve_all["profit_per_10k"]

    # ---- challenger gap on test
    p_ch = challenger.predict(test[challenger.feature_name()])
    gap = pd.DataFrame(
        {
            "LightGBM, all 10, monotone": paired_bootstrap_auc_gap(
                y_te, pd_te, p_ch, ev["n_bootstrap"], seed
            )
        }
    ).T
    gap.insert(0, "challenger_auc", roc_auc_score(y_te, p_ch))

    # ---- R-1: the 96/98 rows
    special = special_code_rows(test)
    r1 = pd.DataFrame(
        {
            "all test rows": summary(y_te, pd_te),
            "without 96/98 rows": summary(y_te[~special], pd_te[~special]),
        }
    )
    sp_group = pd.DataFrame(
        {
            "rows": [int(special.sum())],
            "bad_rate": [float(y_te[special].mean())] if special.any() else [np.nan],
            "mean_pd": [float(pd_te[special].mean())] if special.any() else [np.nan],
            "declined_share": [float((s_te[special] < cut).mean())] if special.any() else [np.nan],
        },
        index=pd.Index(["96/98 rows on test"], name="group"),
    )

    # ---- reason codes on test declines
    declined = test[s_te < cut]
    freq = reason_frequency(
        top_reasons(sc, declined, reference, rc["max_reasons"], rc["min_points_lost"]),
        rc["max_reasons"],
    )

    # ---- report
    pct = ["predicted_pd", "observed_bad_rate"]
    pol_pct = ["approval_rate", "bad_rate_approved", "bads_declined_share", "goods_declined_share"]
    prints_tbl = pd.DataFrame(
        {"sha256 (first 16)": {k: v[:16] for k, v in prints.items()}}
    ).rename_axis("frozen input")
    nl = "\n"
    sections = [
        "# Test-set evaluation (one time)" + nl,
        f"Scored at {run_at} UTC ({runs_note}). Every artifact was fitted on "
        "train and chosen on valid before this run. "
        "Generated by `python -m scorecard.pipelines.evaluate`." + nl,
        "## Frozen inputs" + nl,
        md_table(prints_tbl),
        nl + "## Headline: valid (used for choices) vs test (never seen)" + nl,
        "Deployed PD: integer score -> master scale (D-016)." + nl,
        md_table(head, pct_cols=[]),
        nl + f"95% bootstrap intervals on test ({ev['n_bootstrap']} resamples, seed {seed}):" + nl,
        md_table(ci),
        nl + "## Calibration by score decile (test)" + nl,
        md_table(calib, pct_cols=pct),
        nl + f"## Policy at the frozen cutoff {cut} (D-018)" + nl,
        f"Profit is **Illustrative**, USD per {PER_N:,} applicants." + nl,
        md_table(policy, pct_cols=pol_pct),
        f"{nl}Value over approve-all on test: **${value:,.0f} per {PER_N:,} applicants "
        "(Illustrative)**." + nl,
        "## Challenger gap on test (D-017)" + nl,
        md_table(gap),
        nl + "## Risk R-1: rows with 96/98 past-due codes" + nl,
        md_table(sp_group, pct_cols=["bad_rate", "mean_pd", "declined_share"]),
        "",
        md_table(r1, pct_cols=[]),
        nl + f"## Reason codes on test declines ({len(declined):,} rows)" + nl,
        md_table(freq, pct_cols=["share_as_reason_1", f"share_in_top_{rc['max_reasons']}"]),
    ]
    report = resolve_path(cfg["reports"]["test_evaluation"])
    if not args.rehearse:
        report.write_text(nl.join(sections) + nl, encoding="utf-8")
        record_run(lock_path, prints)  # only now: a crash above is not a look at test

    pd.set_option("display.width", 200)
    print(runs_note, nl)
    print(head.round(5).to_string(), nl)
    print(ci.round(4).to_string(), nl)
    print(calib.round(4).to_string(), nl)
    print(policy.round(4).to_string(), nl, f"value over approve-all: ${value:,.0f}", nl)
    print(gap.round(4).to_string(), nl)
    print(sp_group.round(4).to_string(), nl)
    print(r1.round(5).to_string(), nl)
    print(freq.round(4).to_string(), nl)
    print("Rehearsal: nothing written." if args.rehearse else f"Report: {report}")


if __name__ == "__main__":
    main()
