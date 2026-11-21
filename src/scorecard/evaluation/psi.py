"""Population stability: has the population the model scores moved? (D-021)

    PSI = sum over bands of (actual% - expected%) * ln(actual% / expected%)

"expected" is train (the development sample), "actual" is a recent population.
PSI on the SCORE says whether the model's input mix moved; the same formula on
each feature (CSI, characteristic stability index) says WHICH feature moved.
Outcomes take two years to mature, so this is the early warning.

The bands are frozen from train in a baseline file, so a monitoring job months
later compares like with like and never needs the training data.

    python -m scorecard.evaluation.psi

builds the baseline from train, checks valid and test, and runs simulated drift
scenarios (Illustrative) on valid. Does not change any model.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.split import load_split
from scorecard.evaluation.cutoff import economics_from_config, policy_cutoff
from scorecard.explain.reason_codes import applicant_points
from scorecard.features.binning import assign_bins, missing_index
from scorecard.models.scorecard import Scorecard

BASELINE_VERSION = 1


# --------------------------------------------------------------------------- #
# The index itself
# --------------------------------------------------------------------------- #
def shares(counts: ArrayLike, floor: float) -> np.ndarray:
    """Counts -> shares, with empty bands lifted to `floor` so ln() is defined."""
    c = np.asarray(counts, dtype=float)
    return np.clip(c / c.sum(), floor, None)


def psi_terms(expected: ArrayLike, actual: ArrayLike) -> np.ndarray:
    """Per-band contributions (a - e) * ln(a / e). Each term is >= 0: when a > e
    both factors are positive, when a < e both are negative."""
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    return (a - e) * np.log(a / e)


def psi(expected: ArrayLike, actual: ArrayLike) -> float:
    return float(psi_terms(expected, actual).sum())


def status(value: float, investigate: float, act: float) -> str:
    if value < investigate:
        return "stable"
    return "investigate" if value < act else "ACT"


_SEVERITY = {"stable": 0, "investigate": 1, "ACT": 2}


def worst(*statuses: str) -> str:
    """Overall status = the most severe signal. Score PSI alone can miss a shift
    that one feature's CSI or the approval rate shows plainly (D-021)."""
    return max(statuses, key=_SEVERITY.__getitem__)


# --------------------------------------------------------------------------- #
# Frozen baseline
# --------------------------------------------------------------------------- #
def score_edges(train_scores: ArrayLike, n_bands: int) -> list[float]:
    """Inner edges at train score quantiles; bands are (edge[i-1], edge[i]].
    Integer scores tie, so duplicate quantiles collapse into one edge."""
    q = np.quantile(np.asarray(train_scores, dtype=float), np.linspace(0, 1, n_bands + 1)[1:-1])
    return [float(v) for v in np.unique(q)]


def band_counts(scores: ArrayLike, edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges), np.asarray(scores, dtype=float), side="left")
    return np.bincount(idx, minlength=len(edges) + 1)


def feature_counts(sc: Scorecard, X: pd.DataFrame, feature: str) -> np.ndarray:
    """Counts per scorecard bin (Missing last): the bins that actually move points."""
    edges = sc.encoder_.features_[feature].edges
    idx = assign_bins(X[feature].to_numpy(dtype=float), edges)
    return np.bincount(idx, minlength=missing_index(edges) + 1)


@dataclass(frozen=True)
class Baseline:
    score_edges: list[float]
    score_shares: list[float]
    feature_shares: dict[str, list[float]]
    mean_points: dict[str, float]
    rows: int

    @classmethod
    def from_train(cls, sc: Scorecard, train: pd.DataFrame, n_bands: int, floor: float) -> Baseline:
        s = sc.score(train)
        edges = score_edges(s, n_bands)
        pts, _ = applicant_points(sc, train)
        return cls(
            score_edges=edges,
            score_shares=shares(band_counts(s, edges), floor).tolist(),
            feature_shares={
                f: shares(feature_counts(sc, train, f), floor).tolist() for f in sc.features
            },
            mean_points={f: float(v) for f, v in pts.mean().items()},
            rows=len(train),
        )

    def save(self, path: Path | str) -> None:
        payload = {"schema_version": BASELINE_VERSION, **asdict(self)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Baseline:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("schema_version", None) != BASELINE_VERSION:
            raise ValueError("Unsupported monitoring baseline version")
        return cls(**payload)


# --------------------------------------------------------------------------- #
# Comparing a population with the baseline
# --------------------------------------------------------------------------- #
def score_psi_table(base: Baseline, scores: ArrayLike, floor: float) -> pd.DataFrame:
    """Per-band expected vs actual shares and PSI contribution."""
    actual = shares(band_counts(scores, base.score_edges), floor)
    bounds = [-np.inf, *base.score_edges, np.inf]
    labels = [f"({lo:g}, {hi:g}]" for lo, hi in itertools.pairwise(bounds)]
    return pd.DataFrame(
        {
            "expected": base.score_shares,
            "actual": actual,
            "psi_term": psi_terms(base.score_shares, actual),
        },
        index=pd.Index(labels, name="score band"),
    )


def csi_table(sc: Scorecard, base: Baseline, X: pd.DataFrame, floor: float) -> pd.DataFrame:
    """CSI per feature, and how many score points the shift moved on average.

    CSI says a feature's mix changed; the points shift says whether that change
    matters to the score (a big shift between bins with similar points does not).
    """
    pts, _ = applicant_points(sc, X)
    rows = {}
    for f in sc.features:
        actual = shares(feature_counts(sc, X, f), floor)
        rows[f] = {
            "csi": psi(base.feature_shares[f], actual),
            "mean_points_shift": float(pts[f].mean() - base.mean_points[f]),
        }
    return pd.DataFrame(rows).T.rename_axis("feature").sort_values("csi", ascending=False)


# --------------------------------------------------------------------------- #
# Simulated drift (Illustrative): shows the monitor reacts, and to the right thing
# --------------------------------------------------------------------------- #
def stress_utilisation(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Downturn: everyone carries 30% more revolving balance (relative)."""
    out = df.copy()
    out["RevolvingUtilizationOfUnsecuredLines"] *= 1.3
    return out


def younger_mix(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Marketing shift: applicants under 35 are twice as likely to apply."""
    w = np.where(df["age"] < 35, 2.0, 1.0)
    idx = rng.choice(len(df), size=len(df), replace=True, p=w / w.sum())
    return df.iloc[idx].reset_index(drop=True)


def more_late_payments(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Early delinquency rises: 10% of applicants gain one 30-59 day late payment."""
    out = df.copy()
    col = "NumberOfTime30-59DaysPastDueNotWorse"
    hit = rng.uniform(size=len(out)) < 0.10
    out.loc[hit, col] = out.loc[hit, col] + 1  # NaN (96/98 rows) stays NaN
    return out


SCENARIOS: dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = {
    "utilisation +30%": stress_utilisation,
    "under-35s apply 2x": younger_mix,
    "10% gain a 30-59 late": more_late_payments,
}

MONITORING_PLAN = pd.DataFrame(
    [
        [
            "Score PSI vs baseline",
            "monthly",
            "< 0.10",
            "0.10-0.25: CSI to find the cause",
            "> 0.25: model review",
        ],
        [
            "CSI per feature",
            "monthly",
            "< 0.10",
            "check data feed and mix",
            "> 0.25: review that feature's bins",
        ],
        [
            "Approval rate at cutoff",
            "monthly",
            "within 3 pts of train (87%)",
            "3-6 pts: check PSI, CSI and policy",
            "> 6 pts: credit committee",
        ],
        [
            "Calibration (predicted vs observed bad rate)",
            "quarterly, once outcomes mature (2 yr)",
            "ratio 0.8-1.25",
            "recalibrate master scale",
            "redevelop",
        ],
        [
            "AUC / Gini on matured accounts",
            "quarterly",
            "Gini > 0.65",
            "investigate",
            "Gini < 0.60: redevelop",
        ],
        [
            "Challenger gap (shadow LightGBM)",
            "quarterly",
            "<= 0.015 AUC",
            "check what the challenger learned",
            "> 0.03: redevelop",
        ],
        [
            "Input data: missing rate, 96/98 codes, out-of-range",
            "every batch",
            "as in training",
            "fix feed",
            "stop scoring",
        ],
    ],
    columns=["metric", "frequency", "green", "amber action", "red action"],
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    mon = cfg["monitoring"]
    floor, inv, act = mon["share_floor"], mon["investigate"], mon["act"]
    a_inv, a_act = mon["approval_investigate"], mon["approval_act"]
    sc = Scorecard.from_json(resolve_path(cfg["artifacts"]["scorecard"]))
    cut = policy_cutoff(sc, economics_from_config(cfg))
    interim = resolve_path(cfg["data"]["interim_dir"])
    train, valid, test = (load_split(n, interim) for n in ("train", "valid", "test"))

    base = Baseline.from_train(sc, train, mon["score_bands"], floor)
    base.save(resolve_path(cfg["artifacts"]["monitoring_baseline"]))
    approve_base = float((sc.score(train) >= cut).mean())

    def check(name: str, df: pd.DataFrame) -> dict[str, object]:
        s = sc.score(df)
        p = float(score_psi_table(base, s, floor)["psi_term"].sum())
        c = csi_table(sc, base, df, floor)
        approval = float((s >= cut).mean())
        signals = {
            "score_psi_status": status(p, inv, act),
            "csi_status": status(float(c["csi"].iloc[0]), inv, act),
            "approval_status": status(abs(approval - approve_base), a_inv, a_act),
        }
        return {
            "population": name,
            "score_psi": p,
            "max_csi": float(c["csi"].iloc[0]),
            "max_csi_feature": c.index[0],
            "mean_score_shift": float(s.mean() - sc.score(train).mean()),
            "approval_at_cutoff": approval,
            **signals,
            "OVERALL": worst(*signals.values()),
        }

    real = pd.DataFrame([check("valid", valid), check("test", test)]).set_index("population")
    detail_test = score_psi_table(base, sc.score(test), floor)
    csi_test = csi_table(sc, base, test, floor)

    rng = np.random.default_rng(cfg["split"]["seed"])
    drifted = {name: fn(valid, rng) for name, fn in SCENARIOS.items()}
    sims = pd.DataFrame([check(n, d) for n, d in drifted.items()]).set_index("population")
    csi_sims = pd.DataFrame({n: csi_table(sc, base, d, floor)["csi"] for n, d in drifted.items()})

    pct = ["expected", "actual"]
    nl = "\n"
    sections = [
        "# Monitoring: population stability" + nl,
        "Baseline = **train** scores and feature bins, frozen in "
        f"`{cfg['artifacts']['monitoring_baseline']}`. Thresholds (**Benchmark**, industry rule "
        f"of thumb): PSI < {inv} stable, {inv}-{act} investigate, > {act} act. "
        f"Approval drift (**Assumption**): > {a_inv:.0%} investigate, > {a_act:.0%} act. "
        "OVERALL = the worst of the three signals. "
        "Generated by `python -m scorecard.evaluation.psi`." + nl,
        f"Approval rate on train at cutoff {cut}: {approve_base:.1%}." + nl,
        "## Real populations vs baseline" + nl,
        md_table(real, pct_cols=["approval_at_cutoff"]),
        nl + "### Score bands, test vs train" + nl,
        md_table(detail_test, pct_cols=pct),
        nl + "### CSI per feature, test vs train" + nl,
        md_table(csi_test),
        nl + "## Simulated drift on valid (**Illustrative**)" + nl,
        "Each scenario alters valid in one known way, to show the monitor fires and that "
        "CSI names the feature that moved. Not a forecast." + nl,
        md_table(sims, pct_cols=["approval_at_cutoff"]),
        nl + "### CSI per feature under each scenario" + nl,
        md_table(csi_sims),
        nl + "## Monitoring plan" + nl,
        "Thresholds other than PSI are **Assumptions** for this demo." + nl,
        md_table(MONITORING_PLAN, index=False),
    ]
    report = resolve_path(cfg["reports"]["monitoring"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8")

    pd.set_option("display.width", 220)
    print(f"Train approval at {cut}: {approve_base:.4f}{nl}")
    print(real.round(5).to_string(), nl)
    print(detail_test.round(5).to_string(), nl)
    print(csi_test.round(5).to_string(), nl)
    print(sims.round(4).to_string(), nl)
    print(csi_sims.round(4).to_string(), nl)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
