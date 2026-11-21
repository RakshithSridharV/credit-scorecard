"""The points scorecard: WoE + logistic regression, scaled to integer points (D-016).

    logit(PD) = b0 + sum_i b_i * WoE_i              (fitted on train)
    score     = offset + factor * ln(odds good:bad) = offset - factor * logit(PD)
    factor    = PDO / ln 2,   offset = base_score - factor * ln(base_odds)

The score is split into points per feature bin, rounded to integers, so an
applicant's score is just the sum of the points of the bins they fall in.

    python -m scorecard.models.scorecard

fits on train, compares the alternative model variants (CV on train + valid),
saves models/scorecard.json and writes reports/scorecard_model.md. No test data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import FEATURES, TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.calibration import (
    ScoreToPD,
    calibration_in_the_large,
    calibration_table,
)
from scorecard.evaluation.metrics import summary
from scorecard.features.binning import assign_bins, bin_labels
from scorecard.features.encoder import WoEEncoder

SCHEMA_VERSION = 1
Folds = list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Scaling:
    """Score scale. These are conventions, not estimates (D-016)."""

    base_score: float = 600  # the score at which odds are base_odds
    base_odds: float = 50  # good:bad odds at base_score
    pdo: float = 20  # points to double the odds

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def pd_from_score(self, score: ArrayLike) -> np.ndarray:
        """Invert the scale: score -> odds good:bad -> PD."""
        odds_good = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1 / (1 + odds_good)


class Scorecard:
    """WoE encoder + unpenalised logistic regression + integer points."""

    def __init__(
        self, features: Sequence[str], binning: dict[str, Any], scaling: Scaling | None = None
    ):
        self.features = list(features)
        self.binning = dict(binning)
        self.scaling = scaling or Scaling()

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Scorecard:
        self.encoder_ = WoEEncoder(**self.binning).fit(X[self.features], y)
        woe = self.encoder_.transform(X[self.features])
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(woe, y)
        self.intercept_ = float(lr.intercept_[0])
        self.coef_ = dict(zip(self.features, map(float, lr.coef_[0]), strict=True))
        self.points_ = self._points()
        self.coef_table_ = self._coef_table(woe, np.asarray(y, dtype=float))
        return self

    def _points(self) -> dict[str, list[int]]:
        """Integer points per bin (numeric bins, then Missing) for each feature.

        The intercept and offset are shared equally across features so that
        every attribute's points stand on their own and simply add up.
        """
        n = len(self.features)
        f, off = self.scaling.factor, self.scaling.offset
        return {
            feat: [
                round(-(self.coef_[feat] * w + self.intercept_ / n) * f + off / n)
                for w in self.encoder_.features_[feat].woe
            ]
            for feat in self.features
        }

    def _coef_table(self, woe: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Coefficients with Wald standard errors from the Fisher information.

        cov(b) = (X' W X)^-1 with W = p(1 - p): the standard maximum-likelihood
        result (what statsmodels would print), computed directly with numpy.
        """
        X = np.column_stack([np.ones(len(woe)), woe.to_numpy()])
        b = np.array([self.intercept_, *self.coef_.values()])
        p = 1 / (1 + np.exp(-X @ b))
        cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
        se = np.sqrt(np.diag(cov))
        z = b / se
        return pd.DataFrame(
            {"coef": b, "std_err": se, "z": z, "p_value": 2 * norm.sf(np.abs(z))},
            index=pd.Index(["(intercept)", *self.features], name="term"),
        )

    # ------------------------------------------------------------------ scoring
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Exact model PD, before rounding to points."""
        woe = self.encoder_.transform(X[self.features])
        logit = self.intercept_ + sum(self.coef_[f] * woe[f] for f in self.features)
        return 1 / (1 + np.exp(-np.asarray(logit)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Integer score: the sum of each feature's bin points."""
        total = np.zeros(len(X), dtype=int)
        for f in self.features:
            idx = assign_bins(X[f].to_numpy(dtype=float), self.encoder_.features_[f].edges)
            total += np.asarray(self.points_[f])[idx]
        return total

    def calibrate(self, oof_scores: ArrayLike, y: ArrayLike, pd_floor: float) -> Scorecard:
        """Learn the score -> PD table from out-of-fold train scores."""
        self.score_to_pd_ = ScoreToPD.fit(oof_scores, y, pd_floor)
        return self

    def score_pd(self, X: pd.DataFrame) -> np.ndarray:
        """PD for each applicant, a function of the integer score only, so the
        same score always means the same PD. Uses the calibrated table when
        fitted, otherwise the raw scale."""
        scores = self.score(X)
        table = getattr(self, "score_to_pd_", None)
        return table.predict(scores) if table else self.scaling.pd_from_score(scores)

    def points_table(self) -> pd.DataFrame:
        rows = []
        for f in self.features:
            fb = self.encoder_.features_[f]
            for label, w, pts, g, b in zip(
                bin_labels(fb.edges), fb.woe, self.points_[f], fb.goods, fb.bads, strict=True
            ):
                if g + b > 0:  # skip a Missing bin no train row fell in
                    rows.append({"feature": f, "bin": label, "woe": w, "points": pts})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ persistence
    def to_json(self, path: Path | str) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "features": self.features,
            "binning": self.binning,
            "scaling": asdict(self.scaling),
            "intercept": self.intercept_,
            "coefficients": self.coef_,
            "points": self.points_,
            "encoder": self.encoder_.to_dict(),  # self-contained: one file for the API
            "score_to_pd": asdict(self.score_to_pd_) if hasattr(self, "score_to_pd_") else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> Scorecard:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported scorecard version {payload.get('schema_version')}")
        sc = cls(payload["features"], payload["binning"], Scaling(**payload["scaling"]))
        sc.encoder_ = WoEEncoder.from_dict(payload["encoder"])
        sc.intercept_ = payload["intercept"]
        sc.coef_ = payload["coefficients"]
        sc.points_ = payload["points"]
        if payload.get("score_to_pd"):
            t = payload["score_to_pd"]
            sc.score_to_pd_ = ScoreToPD(tuple(t["scores"]), tuple(t["pds"]))
        return sc


def oof_scores(
    train: pd.DataFrame,
    features: list[str],
    binning: dict[str, Any],
    scaling: Scaling,
    folds: Folds,
) -> np.ndarray:
    """Each train row scored by a scorecard that never saw it (5 fold models)."""
    out = np.empty(len(train), dtype=int)
    for fit_idx, eval_idx in folds:
        part = train.iloc[fit_idx]
        sc = Scorecard(features, binning, scaling).fit(part, part[TARGET])
        out[eval_idx] = sc.score(train.iloc[eval_idx])
    return out


# --------------------------------------------------------------------------- #
# Variant comparison: why this model and not another
# --------------------------------------------------------------------------- #
def woe_lr(binning: dict[str, Any], **lr_kwargs: Any) -> Callable[[], Pipeline]:
    def make() -> Pipeline:
        lr = LogisticRegression(max_iter=1000, **{"penalty": None, **lr_kwargs})
        return Pipeline([("woe", WoEEncoder(**binning)), ("lr", lr)])

    return make


def raw_lr() -> Pipeline:
    """Baseline without WoE: impute medians (+ missing flags), standardise, fit."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(penalty=None, max_iter=5000)),
        ]
    )


def compare_variants(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    selected: list[str],
    binning: dict[str, Any],
    folds: Folds,
) -> pd.DataFrame:
    variants: dict[str, tuple[list[str], Callable[[], Pipeline]]] = {
        "WoE LR, 7 selected (chosen)": (selected, woe_lr(binning)),
        "WoE LR, all 10": (FEATURES, woe_lr(binning)),
        "WoE LR, 7, L2 C=0.01": (selected, woe_lr(binning, penalty="l2", C=0.01)),
        "WoE LR, 7, L2 C=0.001": (selected, woe_lr(binning, penalty="l2", C=0.001)),
        "WoE LR, 7, class_weight=balanced": (
            selected,
            woe_lr(binning, class_weight="balanced"),
        ),
        "Raw LR, all 10 (no WoE)": (FEATURES, raw_lr),
    }
    rows = []
    for name, (cols, make) in variants.items():
        cv = cross_val_score(make(), train[cols], train[TARGET], cv=folds, scoring="roc_auc")
        model = make().fit(train[cols], train[TARGET])
        m = summary(valid[TARGET], model.predict_proba(valid[cols])[:, 1])
        rows.append(
            {
                "variant": name,
                "cv_auc": cv.mean(),
                "cv_sd": cv.std(ddof=1),
                "valid_auc": m["auc"],
                "valid_gini": m["gini"],
                "valid_ks": m["ks"],
                "valid_brier": m["brier"],
                "mean_pd": m["mean_pd"],
            }
        )
    return pd.DataFrame(rows).set_index("variant")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    interim = resolve_path(cfg["data"]["interim_dir"])
    train, valid = load_split("train", interim), load_split("valid", interim)
    selected = json.loads(
        resolve_path(cfg["artifacts"]["selected_features"]).read_text(encoding="utf-8")
    )["features"]
    binning, sc_cfg = cfg["binning"], dict(cfg["scorecard"])
    pd_floor = sc_cfg.pop("pd_floor")
    scaling = Scaling(**sc_cfg)
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=cfg["split"]["seed"]).split(
            train, train[TARGET]
        )
    )

    variants = compare_variants(train, valid, selected, binning, folds)

    sc = Scorecard(selected, binning, scaling).fit(train, train[TARGET])
    raw_pd_valid = sc.score_pd(valid)  # before calibration: PD from the raw scale
    sc.calibrate(oof_scores(train, selected, binning, scaling, folds), train[TARGET], pd_floor)
    sc.to_json(resolve_path(cfg["artifacts"]["scorecard"]))

    y_va, scores = valid[TARGET], sc.score(valid)
    pd_va = sc.score_pd(valid)
    metrics = pd.DataFrame(
        {
            "exact_pd": summary(y_va, sc.predict_pd(valid)),
            "points_raw_scale": summary(y_va, raw_pd_valid),
            "points_calibrated": summary(y_va, pd_va),
        }
    )
    metrics.loc["auc_of_score"] = roc_auc_score(y_va, -scores)  # ranking by the score itself
    metrics.loc["mean_pd / bad_rate"] = [
        calibration_in_the_large(y_va, metrics_pd)
        for metrics_pd in (sc.predict_pd(valid), raw_pd_valid, pd_va)
    ]

    def by_score_decile(p: np.ndarray) -> pd.DataFrame:
        t = calibration_table(y_va, p, rank_by=-scores)
        decile = pd.qcut(pd.Series(-scores).rank(method="first"), 10, labels=False).to_numpy()
        grouped = pd.Series(scores).groupby(decile)
        t["score_min"], t["score_max"] = grouped.min().to_numpy(), grouped.max().to_numpy()
        return t

    calib_raw, calib = by_score_decile(raw_pd_valid), by_score_decile(pd_va)
    master = pd.DataFrame({"score": sc.score_to_pd_.scores, "pd": sc.score_to_pd_.pds})

    s = sc.scaling
    pct = ["predicted_pd", "observed_bad_rate"]
    nl = "\n"
    sections = [
        "# Scorecard model" + nl,
        "Fitted on **train**; metrics are on **valid** unless marked CV (5-fold on train). "
        "Test is not used. Generated by `python -m scorecard.models.scorecard`." + nl,
        "## Variants compared (D-016)" + nl,
        md_table(variants, pct_cols=["mean_pd"]),
        f"{nl}Valid bad rate: {y_va.mean():.2%}.{nl}",
        "## Chosen model: coefficients" + nl,
        "All coefficients should be negative (WoE positive = safer). "
        "Standard errors from the Fisher information." + nl,
        md_table(sc.coef_table_),
        nl + "## Scaling (convention)" + nl,
        f"Base score {s.base_score:g} at odds {s.base_odds:g}:1 good:bad, PDO {s.pdo:g}. "
        f"Factor = {s.factor:.4f}, offset = {s.offset:.4f}." + nl,
        "## Scorecard (points per bin)" + nl,
        md_table(sc.points_table(), index=False),
        nl + "## Metrics on valid" + nl,
        md_table(metrics),
        nl + "## Calibration by score decile, raw scale (before)" + nl,
        md_table(calib_raw, pct_cols=pct),
        nl + "## Calibration by score decile, calibrated (after)" + nl,
        "Score -> PD table fitted by isotonic regression on out-of-fold train scores, "
        f"PD floor {pd_floor:.2%}." + nl,
        md_table(calib, pct_cols=pct),
        nl + "## Score -> PD table (master scale)" + nl,
        md_table(master, pct_cols=["pd"], index=False),
    ]
    report = resolve_path(cfg["reports"]["scorecard_model"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(variants.round(4).to_string(), nl)
    print(sc.coef_table_.round(4).to_string(), nl)
    print(metrics.round(5).to_string(), nl)
    print("BEFORE", nl, calib_raw.round(4).to_string(), nl, "AFTER", nl, calib.round(4).to_string())
    print(f"{nl}Valid scores: {scores.min()} to {scores.max()}, median {np.median(scores):.0f}")
    print(f"Saved {cfg['artifacts']['scorecard']}; report in {report}")


if __name__ == "__main__":
    main()
