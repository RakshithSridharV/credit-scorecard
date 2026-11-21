"""Calibration: when the model says 10% PD, do about 10% default?

Pricing and loss provisions use the PD itself, not just the ranking, so a
well-ranking but badly calibrated model would misprice every loan.

We do not use the Hosmer-Lemeshow test: with ~20k rows it flags differences
far too small to matter. A decile table plus the overall ratio is clearer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression


def calibration_table(
    y: ArrayLike, p_bad: ArrayLike, n_bins: int = 10, rank_by: ArrayLike | None = None
) -> pd.DataFrame:
    """Predicted vs observed bad rate per decile, lowest risk first.

    Deciles are formed on `rank_by` (default: the PD itself). Pass minus the
    score to group by score, which matters when many scores share one PD.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p_bad, dtype=float)})
    key = pd.Series(df["p"] if rank_by is None else np.asarray(rank_by, dtype=float))
    # rank(method="first") breaks ties, so qcut always finds n_bins equal groups
    # even when a points scorecard gives many applicants the same value.
    df["decile"] = pd.qcut(key.rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    t = df.groupby("decile", observed=True).agg(
        rows=("y", "size"), predicted_pd=("p", "mean"), observed_bad_rate=("y", "mean")
    )
    t["ratio"] = t["predicted_pd"] / t["observed_bad_rate"]
    return t


def calibration_in_the_large(y: ArrayLike, p_bad: ArrayLike) -> float:
    """Mean predicted PD / observed bad rate. 1.0 = right on average."""
    return float(np.mean(p_bad) / np.mean(y))


@dataclass(frozen=True)
class ScoreToPD:
    """Score -> PD lookup ("master scale"), learned by isotonic regression (D-016).

    Isotonic regression fits the best step function that never goes up as the
    score rises, so a higher score can never mean a higher PD. Prediction
    interpolates linearly between the fitted points and clips outside them.
    Fit it on OUT-OF-FOLD scores, never on scores from the model's own
    training rows, or it learns the model's optimism.
    """

    scores: tuple[float, ...]  # ascending
    pds: tuple[float, ...]  # non-increasing

    @classmethod
    def fit(cls, scores: ArrayLike, y: ArrayLike, pd_floor: float) -> ScoreToPD:
        iso = IsotonicRegression(
            increasing=False, y_min=pd_floor, y_max=1 - pd_floor, out_of_bounds="clip"
        ).fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
        return cls(
            tuple(float(s) for s in iso.X_thresholds_), tuple(float(p) for p in iso.y_thresholds_)
        )

    def predict(self, scores: ArrayLike) -> np.ndarray:
        # np.interp clips to the end values outside the fitted range.
        return np.interp(np.asarray(scores, dtype=float), self.scores, self.pds)
