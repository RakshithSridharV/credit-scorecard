"""WoE encoder: learn bins + WoE on train, apply them unchanged anywhere.

Follows the scikit-learn transformer contract (fit / transform, settings in
__init__, learned state in attributes ending in "_"), so it can sit in a
Pipeline later. The fitted state is saved as JSON, not pickle: readable,
diffable, and safe to load in the scoring API.

    python -m scorecard.features.encoder

fits on data/interim/train.csv, saves the encoder to models/, and writes
reports/woe_summary.md with train IV next to valid IV. Test is not touched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import FEATURES, TARGET
from scorecard.data.split import load_split
from scorecard.features.binning import assign_bins, bin_counts, bin_labels, coarse_edges
from scorecard.features.iv import information_value, iv_strength
from scorecard.features.woe import woe_values

SCHEMA_VERSION = 1  # bump if the JSON layout changes, so old files fail loudly


@dataclass(frozen=True)
class FeatureBins:
    """Everything learned for one feature. Lists run numeric bins first, Missing last."""

    edges: tuple[float, ...]
    direction: str
    woe: tuple[float, ...]
    goods: tuple[float, ...]  # train counts, kept for the report and for audit
    bads: tuple[float, ...]


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replace each feature value by the WoE of the train bin it falls in."""

    def __init__(
        self,
        fine_bins: int = 20,
        min_bin_share: float = 0.02,
        max_bins: int = 6,
        smoothing: float = 0.5,
    ):
        self.fine_bins = fine_bins
        self.min_bin_share = min_bin_share
        self.max_bins = max_bins
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoEEncoder:
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_arr)}")
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("y must contain only 0 (good) and 1 (bad)")

        self.features_: dict[str, FeatureBins] = {}
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            edges, direction = coarse_edges(
                x,
                y_arr,
                fine_bins=self.fine_bins,
                min_bin_share=self.min_bin_share,
                max_bins=self.max_bins,
                smoothing=self.smoothing,
            )
            goods, bads = bin_counts(x, y_arr, edges)
            self.features_[col] = FeatureBins(
                edges=tuple(float(e) for e in edges),
                direction=direction,
                woe=tuple(float(w) for w in woe_values(goods, bads, self.smoothing)),
                goods=tuple(float(g) for g in goods),
                bads=tuple(float(b) for b in bads),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """WoE values, same index and feature columns. Needs no target.

        A NaN in a feature that had no missing values in train gets WoE 0
        (no evidence); the API rejects such input before it gets here.
        """
        check_is_fitted(self, "features_")
        missing = set(self.features_) - set(X.columns)
        if missing:
            raise ValueError(f"Cannot transform: missing columns {sorted(missing)}")
        return pd.DataFrame(
            {
                col: np.asarray(fb.woe)[assign_bins(X[col].to_numpy(dtype=float), fb.edges)]
                for col, fb in self.features_.items()
            },
            index=X.index,
        )

    # ----------------------------------------------------------------- reporting
    def iv(self) -> pd.Series:
        """Train IV per feature, highest first."""
        check_is_fitted(self, "features_")
        iv = {
            c: information_value(fb.goods, fb.bads, self.smoothing)
            for c, fb in self.features_.items()
        }
        return pd.Series(iv, name="iv").sort_values(ascending=False)

    def bin_table(self, feature: str, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rows, bad rate and IV contribution per fitted bin, on any data set."""
        check_is_fitted(self, "features_")
        fb = self.features_[feature]
        goods, bads = bin_counts(X[feature].to_numpy(dtype=float), y, fb.edges)
        rows = goods + bads
        t = pd.DataFrame(
            {
                "rows": rows,
                "share": rows / rows.sum(),
                "bad_rate": np.divide(bads, rows, out=np.full_like(rows, np.nan), where=rows > 0),
                "woe": fb.woe,  # always the TRAIN WoE: that is what the model will see
            },
            index=pd.Index(bin_labels(fb.edges), name="bin"),
        )
        t["iv_part"] = (goods / goods.sum() - bads / bads.sum()) * woe_values(
            goods, bads, self.smoothing
        )
        return t

    # ----------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """Plain-Python state, so other artifacts (the scorecard) can embed it."""
        check_is_fitted(self, "features_")
        return {
            "schema_version": SCHEMA_VERSION,
            "params": self.get_params(),
            "features": {c: asdict(fb) for c, fb in self.features_.items()},
        }

    def to_json(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> WoEEncoder:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WoEEncoder:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported encoder file version {payload.get('schema_version')}")
        enc = cls(**payload["params"])
        enc.features_ = {
            c: FeatureBins(**{k: v if k == "direction" else tuple(v) for k, v in fb.items()})
            for c, fb in payload["features"].items()
        }
        return enc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def iv_summary(enc: WoEEncoder, valid: pd.DataFrame) -> pd.DataFrame:
    """Train vs valid IV per feature, using the TRAIN bins on both."""
    train_iv = enc.iv()
    rows = []
    for col in train_iv.index:
        fb = enc.features_[col]
        goods, bads = bin_counts(valid[col].to_numpy(dtype=float), valid[TARGET], fb.edges)
        rows.append(
            {
                "feature": col,
                "bins": len(fb.edges) + 1,
                "direction": fb.direction,
                "iv_train": train_iv[col],
                "iv_valid": information_value(goods, bads, enc.smoothing),
                "strength": iv_strength(train_iv[col]),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def render_report(enc: WoEEncoder, train: pd.DataFrame, valid: pd.DataFrame) -> str:
    summary = iv_summary(enc, valid)
    parts = [
        "# WoE binning summary\n",
        (
            "Bins and WoE fitted on **train only** (D-002, D-014); valid columns use the same "
            "train bins. Test is not used. Generated by `python -m scorecard.features.encoder`.\n"
        ),
        (
            f"Settings: fine_bins={enc.fine_bins}, min_bin_share={enc.min_bin_share:.0%}, max_bins={enc.max_bins}, "
            f"smoothing={enc.smoothing}. WoE = ln(%good / %bad): positive = safer.\n"
        ),
        "## Information Value\n",
        md_table(summary),
    ]
    for col in summary.index:
        tr = enc.bin_table(col, train, train[TARGET])
        va = enc.bin_table(col, valid, valid[TARGET])
        t = tr[["rows", "share", "bad_rate", "woe", "iv_part"]].copy()
        t["valid_share"] = va["share"]
        t["valid_bad_rate"] = va["bad_rate"]
        t = t[(tr["rows"] > 0) | (va["rows"] > 0)]  # hide an empty Missing bin
        parts += [
            f"\n## {col}\n",
            (
                f"Bad rate {enc.features_[col].direction} with the value. "
                f"Train IV {summary.loc[col, 'iv_train']:.3f}, valid IV "
                f"{summary.loc[col, 'iv_valid']:.3f}.\n"
            ),
            md_table(t, pct_cols=["share", "bad_rate", "valid_share", "valid_bad_rate"]),
        ]
    return "\n".join(parts) + "\n"


def main() -> None:
    cfg = load_config()
    interim = resolve_path(cfg["data"]["interim_dir"])
    train = load_split("train", interim)
    valid = load_split("valid", interim)

    enc = WoEEncoder(**cfg["binning"]).fit(train[FEATURES], train[TARGET])

    artifact = resolve_path(cfg["artifacts"]["woe_encoder"])
    enc.to_json(artifact)
    report = resolve_path(cfg["reports"]["woe_summary"])
    report.write_text(render_report(enc, train, valid), encoding="utf-8")

    print(iv_summary(enc, valid).round(3).to_string())
    print(f"\nSaved encoder to {artifact}; report in {report}")


if __name__ == "__main__":
    main()
