"""Discrimination and accuracy metrics for a PD model.

Discrimination (does the model RANK bads above goods?): AUC, Gini, KS.
Accuracy of the probabilities themselves: Brier score, log loss.
A model can rank perfectly and still give wrong PDs, so we need both kinds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def gini(y: ArrayLike, p_bad: ArrayLike) -> float:
    """Gini = 2 * AUC - 1: the AUC rescaled so random = 0 and perfect = 1.
    Banks usually quote Gini; it is the same information as AUC."""
    return 2 * roc_auc_score(y, p_bad) - 1


def ks(y: ArrayLike, p_bad: ArrayLike) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative score
    distributions of bads and goods, i.e. max over cutoffs of (TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y, p_bad)
    return float(np.max(tpr - fpr))


def summary(y: ArrayLike, p_bad: ArrayLike) -> dict[str, float]:
    """All headline metrics for one data set."""
    y_arr = np.asarray(y, dtype=float)
    p = np.asarray(p_bad, dtype=float)
    return {
        "auc": float(roc_auc_score(y_arr, p)),
        "gini": gini(y_arr, p),
        "ks": ks(y_arr, p),
        "brier": float(brier_score_loss(y_arr, p)),
        "log_loss": float(log_loss(y_arr, p)),
        "mean_pd": float(p.mean()),
        "bad_rate": float(y_arr.mean()),
    }
