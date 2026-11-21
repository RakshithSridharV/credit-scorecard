"""Feature selection for the scorecard (D-015). Uses TRAIN only.

Three gates, cheapest first:
1. IV filter: drop features too weak to earn a scorecard slot.
2. Redundancy: drop the weaker of any highly correlated pair, then anything
   with a high VIF.
3. Backward elimination by 5-fold cross-validated AUC on train. Each round
   drops a feature with the wrong coefficient sign if there is one, otherwise
   the feature whose removal costs the least AUC, as long as that cost is
   below `min_auc_loss`. The WoE encoder is refitted inside every fold, so a
   fold's held-out rows never help learn its bins.

Valid is not used to choose anything here; it stays an honest check for the
model step. Writes models/selected_features.json and reports/feature_selection.md.

    python -m scorecard.features.selection
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import FEATURES, TARGET
from scorecard.data.split import load_split
from scorecard.features.encoder import WoEEncoder

Folds = list[tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------- #
# Gates 1 and 2: filters on the WoE-encoded train data
# --------------------------------------------------------------------------- #
def iv_filter(iv: pd.Series, min_iv: float) -> list[str]:
    """Features with IV >= min_iv, strongest first."""
    return iv[iv >= min_iv].sort_values(ascending=False).index.tolist()


def correlation_filter(iv: pd.Series, corr: pd.DataFrame, max_corr: float) -> list[str]:
    """Walk features from highest IV down; keep one only if it is not too
    correlated with any feature already kept. So of a correlated pair, the
    stronger one survives."""
    kept: list[str] = []
    for f in iv.sort_values(ascending=False).index:
        if all(abs(corr.loc[f, k]) <= max_corr for k in kept):
            kept.append(f)
    return kept


def vif(woe: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor of each column.

    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing column j on all
    the others. Those values are exactly the diagonal of the inverse
    correlation matrix, which avoids fitting one regression per column.
    """
    inv = np.linalg.inv(woe.corr().to_numpy())
    return pd.Series(np.diag(inv), index=woe.columns, name="vif")


def vif_filter(iv: pd.Series, woe: pd.DataFrame, max_vif: float) -> list[str]:
    """While any VIF exceeds max_vif, drop the weakest (lowest IV) offender."""
    cols = list(woe.columns)
    while len(cols) > 1:
        v = vif(woe[cols])
        offenders = v[v > max_vif].index
        if offenders.empty:
            break
        cols.remove(iv[offenders].idxmin())
    return cols


# --------------------------------------------------------------------------- #
# Gate 3: backward elimination by cross-validated AUC
# --------------------------------------------------------------------------- #
def make_model(binning: dict[str, Any]) -> Pipeline:
    """WoE encoder + unpenalised logistic regression.

    No penalty: with about 100k rows a ridge penalty changes almost nothing,
    and unpenalised coefficients are the plain maximum-likelihood estimates,
    which are easier to explain. lbfgs is deterministic, so no seed is needed.
    """
    return Pipeline(
        [
            ("woe", WoEEncoder(**binning)),
            ("lr", LogisticRegression(penalty=None, max_iter=1000)),
        ]
    )


def fold_aucs(
    X: pd.DataFrame, y: pd.Series, cols: Sequence[str], folds: Folds, binning: dict[str, Any]
) -> np.ndarray:
    """Held-out AUC per fold, refitting encoder and model inside each fold."""
    aucs = []
    for fit_idx, eval_idx in folds:
        model = make_model(binning).fit(X.iloc[fit_idx][list(cols)], y.iloc[fit_idx])
        p_bad = model.predict_proba(X.iloc[eval_idx][list(cols)])[:, 1]
        aucs.append(roc_auc_score(y.iloc[eval_idx], p_bad))
    return np.array(aucs)


def wrong_sign_features(coef: Sequence[float], cols: Sequence[str]) -> list[str]:
    """Features whose coefficient is >= 0.

    WoE is positive for SAFER bins and the model predicts the probability of
    BAD, so every coefficient must be negative. A positive one would mean
    "safer bin, higher risk": the feature is fighting another one it overlaps.
    """
    return [c for c, k in zip(cols, coef, strict=True) if k >= 0]


def backward_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    cols: Sequence[str],
    *,
    binning: dict[str, Any],
    folds: Folds,
    min_auc_loss: float,
) -> tuple[list[str], pd.DataFrame]:
    """Drop features one at a time; return the survivors and a round-by-round log."""
    cols = list(cols)
    history = []
    while len(cols) > 1:
        base = fold_aucs(X, y, cols, folds, binning)
        coef = make_model(binning).fit(X[cols], y)[-1].coef_[0]
        wrong = wrong_sign_features(coef, cols)

        # Paired comparison: same folds with and without each feature.
        loss = {}
        for f in cols:
            delta = base - fold_aucs(X, y, [c for c in cols if c != f], folds, binning)
            loss[f] = (float(delta.mean()), int((delta > 0).sum()))

        candidates = wrong or cols  # a wrong sign is removed before anything else
        drop = min(candidates, key=lambda f: loss[f][0])
        if not wrong and loss[drop][0] >= min_auc_loss:
            history.append({"features": len(cols), "cv_auc": base.mean(), "dropped": "(stop)"})
            break
        history.append(
            {
                "features": len(cols),
                "cv_auc": base.mean(),
                "dropped": drop,
                "reason": "wrong sign" if wrong else "AUC loss below tolerance",
                "auc_loss": loss[drop][0],
                "folds_worse_without": f"{loss[drop][1]}/{len(folds)}",
            }
        )
        cols.remove(drop)
    return cols, pd.DataFrame(history).set_index("features")


def drop_one_table(
    X: pd.DataFrame, y: pd.Series, cols: Sequence[str], folds: Folds, binning: dict[str, Any]
) -> pd.DataFrame:
    """For the final set: AUC lost when each feature is removed, plus its coefficient."""
    base = fold_aucs(X, y, cols, folds, binning)
    coef = make_model(binning).fit(X[list(cols)], y)[-1].coef_[0]
    rows = []
    for f, k in zip(cols, coef, strict=True):
        delta = base - fold_aucs(X, y, [c for c in cols if c != f], folds, binning)
        rows.append({"feature": f, "coef": k, "auc_loss_if_dropped": delta.mean()})
    return (
        pd.DataFrame(rows).set_index("feature").sort_values("auc_loss_if_dropped", ascending=False)
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    sel, binning, seed = cfg["selection"], cfg["binning"], cfg["split"]["seed"]
    interim = resolve_path(cfg["data"]["interim_dir"])
    train = load_split("train", interim)
    X, y = train[FEATURES], train[TARGET]

    encoder = WoEEncoder(**binning).fit(X, y)
    iv, woe = encoder.iv(), encoder.transform(X)
    corr = woe.corr()

    after_iv = iv_filter(iv, sel["min_iv"])
    after_corr = correlation_filter(iv[after_iv], corr, sel["max_corr"])
    after_vif = vif_filter(iv, woe[after_corr], sel["max_vif"])

    folds = list(StratifiedKFold(sel["cv_folds"], shuffle=True, random_state=seed).split(X, y))
    selected, history = backward_elimination(
        X, y, after_vif, binning=binning, folds=folds, min_auc_loss=sel["min_auc_loss"]
    )
    final = drop_one_table(X, y, selected, folds, binning)

    # Valid is only REPORTED here, never used to choose.
    valid = load_split("valid", interim)
    valid_auc = {}
    for name, cols in (("all features", FEATURES), ("selected", selected)):
        model = make_model(binning).fit(X[cols], y)
        valid_auc[name] = roc_auc_score(valid[TARGET], model.predict_proba(valid[cols])[:, 1])

    artifact = resolve_path(cfg["artifacts"]["selected_features"])
    artifact.write_text(
        json.dumps({"features": selected, "settings": sel, "seed": seed}, indent=2),
        encoding="utf-8",
    )

    short = {f: f"f{i}" for i, f in enumerate(FEATURES, 1)}
    gates = pd.DataFrame(
        {
            "iv": iv,
            "vif_all": vif(woe),
            "passed_iv": [f in after_iv for f in iv.index],
            "passed_redundancy": [f in after_vif for f in iv.index],
            "selected": [f in selected for f in iv.index],
        }
    )
    report = resolve_path(cfg["reports"]["feature_selection"])
    report.write_text(
        "\n".join(
            [
                "# Feature selection\n",
                (
                    "Train only (D-015). Generated by `python -m scorecard.features.selection`. "
                    f"Settings: {sel}, seed {seed}.\n"
                ),
                f"**Selected ({len(selected)}):** " + ", ".join(f"`{f}`" for f in selected) + "\n",
                "## Gates 1-2: IV and redundancy\n",
                md_table(gates),
                "\n## WoE correlation on train\n",
                "Key: " + ", ".join(f"{v} = `{k}`" for k, v in short.items()) + "\n",
                md_table(corr.rename(index=short, columns=short).round(2)),
                "\n## Gate 3: backward elimination (5-fold CV AUC on train)\n",
                md_table(history),
                "\n## Final set: AUC lost if each feature is removed\n",
                "All coefficients must be negative (WoE positive = safer).\n",
                md_table(final),
                "\n## Valid AUC (reported, not used for selection)\n",
                md_table(pd.Series(valid_auc, name="valid_auc").to_frame()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(history.to_string(), "\n")
    print(final.to_string(), "\n")
    print({k: round(v, 4) for k, v in valid_auc.items()})
    print(f"\nSaved {artifact}; report in {report}")


if __name__ == "__main__":
    main()
