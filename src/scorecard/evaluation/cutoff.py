"""Cutoff strategy: turn the score into approve / decline (D-018).

Approve when score >= cutoff. Each cutoff trades approval rate (volume) against
the bad rate among the approved (losses). With a profit per good loan and a
loss per bad loan, the best rule is to approve an applicant exactly when

    PD * loss_per_bad < (1 - PD) * profit_per_good
    <=>  PD < break_even_pd = profit_per_good / (profit_per_good + loss_per_bad)

so the cutoff comes from the calibrated PD (master scale, fitted on train),
not from maximising profit on one sample. The strategy table on valid checks it.

The money inputs are business assumptions, not data. Every number derived from
them is ILLUSTRATIVE: it shows the method, not the economics of a real lender.

    python -m scorecard.evaluation.cutoff

uses train-fitted artifacts and valid data only. No test data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.calibration import ScoreToPD
from scorecard.models.scorecard import Scorecard

PER_N = 10_000  # money is reported per 10,000 applicants so it reads at a glance


@dataclass(frozen=True)
class Economics:
    """Money per loan. Labels: see configs/config.yaml (Illustrative / Assumption)."""

    ead: float  # exposure at default, USD
    annual_margin: float  # yearly income on a performing loan, as a share of ead
    horizon_years: float  # matches the target window (2 years)
    lgd: float  # loss given default, share of ead

    @property
    def profit_per_good(self) -> float:
        return self.ead * self.annual_margin * self.horizon_years

    @property
    def loss_per_bad(self) -> float:
        # Simplification: no income earned before default, no discounting.
        return self.ead * self.lgd

    @property
    def break_even_pd(self) -> float:
        return self.profit_per_good / (self.profit_per_good + self.loss_per_bad)

    def scale_loss(self, k: float) -> Economics:
        """Same economics with loss per bad multiplied by k (via LGD), for sensitivity."""
        return replace(self, lgd=self.lgd * k)


def strategy_table(scores: ArrayLike, y: ArrayLike, econ: Economics) -> pd.DataFrame:
    """One row per possible cutoff c (approve score >= c), on observed outcomes.

    Sorting by score descending and taking cumulative sums gives every cutoff in
    one pass instead of re-filtering the data for each.
    """
    df = pd.DataFrame({"score": np.asarray(scores), "bad": np.asarray(y, dtype=int)})
    by_score = df.groupby("score")["bad"].agg(n="size", bads="sum").sort_index(ascending=False)
    approved = by_score["n"].cumsum()  # everyone at or above this score
    bads = by_score["bads"].cumsum()
    goods = approved - bads
    total_n, total_bads = len(df), int(df["bad"].sum())
    profit = goods * econ.profit_per_good - bads * econ.loss_per_bad
    t = pd.DataFrame(
        {
            "approval_rate": approved / total_n,
            "bad_rate_approved": bads / approved,
            "bads_declined_share": 1 - bads / total_bads,  # "capture": bads kept out
            "goods_declined_share": 1 - goods / (total_n - total_bads),  # lost business
            "profit_per_10k": profit / total_n * PER_N,
        }
    )
    return t.sort_index().rename_axis("cutoff")


def cutoff_from_pd(table: ScoreToPD, break_even_pd: float, lo: int, hi: int) -> int:
    """Lowest integer score whose master-scale PD is at or below break-even.

    The master scale never rises with the score, so every score at or above
    this one also clears break-even: the rule is a single score cutoff.
    """
    grid = np.arange(lo, hi + 1)
    ok = grid[table.predict(grid) <= break_even_pd]
    if ok.size == 0:
        raise ValueError(f"No score in [{lo}, {hi}] reaches PD <= {break_even_pd:.4f}")
    return int(ok.min())


def economics_from_config(cfg: dict[str, Any]) -> Economics:
    """Money inputs from the `cutoff:` section (sensitivity multipliers left out)."""
    return Economics(**{k: v for k, v in cfg["cutoff"].items() if k != "sensitivity"})


def policy_cutoff(sc: Scorecard, econ: Economics) -> int:
    """The D-018 cutoff for a fitted scorecard: searched over every score the
    scorecard can produce, so it does not depend on any data sample."""
    lo = sum(min(p) for p in sc.points_.values())
    hi = sum(max(p) for p in sc.points_.values())
    return cutoff_from_pd(sc.score_to_pd_, econ.break_even_pd, lo, hi)


def row_at(strategy: pd.DataFrame, cutoff: int) -> pd.Series:
    """Strategy row for a cutoff that may fall between observed scores.

    Approving score >= c is the same as approving score >= the next observed score.
    """
    at_or_above = strategy.index[strategy.index >= cutoff]
    return strategy.loc[at_or_above.min()]


def band_bad_rates(scores: ArrayLike, y: ArrayLike, cutoff: int, width: int = 5) -> pd.DataFrame:
    """Bad rate in narrow score bands around the cutoff: the MARGINAL bad rate.

    The applicants a cutoff move adds or removes sit at the margin, so it is
    their bad rate, not the average of all approved, that should meet break-even.
    """
    s, yy = np.asarray(scores), np.asarray(y, dtype=float)
    rows = []
    for lo in range(cutoff - 4 * width, cutoff + 4 * width, width):
        m = (s >= lo) & (s < lo + width)
        rows.append(
            {
                "band": f"[{lo}, {lo + width})",
                "rows": int(m.sum()),
                "bad_rate": float(yy[m].mean()) if m.any() else np.nan,
                "side": "approve" if lo >= cutoff else "decline",
            }
        )
    return pd.DataFrame(rows).set_index("band")


def sensitivity(
    econ: Economics,
    multipliers: list[float],
    table: ScoreToPD,
    scores: np.ndarray,
    y: ArrayLike,
    strategy: pd.DataFrame,
) -> pd.DataFrame:
    """How the cutoff and its outcomes move if the loss per bad changes."""
    lo, hi = int(scores.min()), int(scores.max())
    rows = []
    for k in [1.0, *multipliers]:
        e = econ.scale_loss(k)
        c = cutoff_from_pd(table, e.break_even_pd, lo, hi)
        own = strategy_table(scores, y, e)  # profit must use this scenario's money
        r = row_at(own, c)
        rows.append(
            {
                "loss_per_bad_x": k,
                "loss_per_bad": e.loss_per_bad,
                "break_even_pd": e.break_even_pd,
                "cutoff": c,
                "approval_rate": r["approval_rate"],
                "bad_rate_approved": r["bad_rate_approved"],
                "profit_per_10k": r["profit_per_10k"],
            }
        )
    return pd.DataFrame(rows).set_index("loss_per_bad_x")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    multipliers = cfg["cutoff"]["sensitivity"]
    econ = economics_from_config(cfg)

    sc = Scorecard.from_json(resolve_path(cfg["artifacts"]["scorecard"]))
    valid = load_split("valid", resolve_path(cfg["data"]["interim_dir"]))
    y, scores = valid[TARGET].to_numpy(), sc.score(valid)

    strategy = strategy_table(scores, y, econ)
    cut = policy_cutoff(sc, econ)
    chosen = row_at(strategy, cut)
    best_on_valid = int(strategy["profit_per_10k"].idxmax())  # in-sample optimum, reference only

    def at_approval(rate: float) -> int:
        """Highest cutoff that still approves at least `rate` of applicants."""
        return int(strategy.index[strategy["approval_rate"] >= rate].max())

    cutoffs = {
        "approve all": int(strategy.index.min()),
        **{f"top {r:.0%}": at_approval(r) for r in (0.95, 0.9, 0.8, 0.7)},
        "PD rule (chosen)": cut,
        "best on valid (in-sample, reference)": best_on_valid,
    }
    points = pd.DataFrame({name: row_at(strategy, c) for name, c in cutoffs.items()}).T
    points.insert(0, "cutoff", list(cutoffs.values()))

    sens = sensitivity(econ, multipliers, sc.score_to_pd_, scores, y, strategy)
    bands = band_bad_rates(scores, y, cut)
    value = chosen["profit_per_10k"] - strategy.iloc[0]["profit_per_10k"]

    pct = ["approval_rate", "bad_rate_approved", "bads_declined_share", "goods_declined_share"]
    nl = "\n"
    sections = [
        "# Cutoff strategy" + nl,
        "Scores and the score -> PD table come from the train-fitted scorecard (D-016); "
        "outcomes are **valid**. Test is not used. "
        "Generated by `python -m scorecard.evaluation.cutoff`." + nl,
        "## Money inputs" + nl,
        "| input | value | label |",
        "|---|---|---|",
        f"| Exposure per account | ${econ.ead:,.0f} | Illustrative |",
        f"| Yearly margin after funding and servicing | {econ.annual_margin:.0%} | Assumption |",
        f"| Horizon | {econ.horizon_years:g} years | Data fact (target window) |",
        f"| Loss given default | {econ.lgd:.0%} | Assumption |",
        f"| Profit per good loan | ${econ.profit_per_good:,.0f} | Illustrative (derived) |",
        f"| Loss per bad loan | ${econ.loss_per_bad:,.0f} | Illustrative (derived) |",
        f"| Break-even PD | {econ.break_even_pd:.2%} | Illustrative (derived) |",
        "",
        "## Operating points (valid)" + nl,
        f"Profit is **Illustrative**, USD per {PER_N:,} applicants.{nl}",
        md_table(points, pct_cols=pct),
        (
            f"{nl}Value of the scorecard over approving everyone at the PD-rule cutoff: "
            f"**${value:,.0f} per {PER_N:,} applicants (Illustrative)**.{nl}"
        ),
        "## Marginal bad rate around the cutoff (valid)" + nl,
        f"Bands of 5 points. Break-even PD = {econ.break_even_pd:.2%}: bands below the "
        "cutoff should sit above it, bands above it below." + nl,
        md_table(bands, pct_cols=["bad_rate"]),
        nl + "## Sensitivity: loss per bad halved / doubled" + nl,
        md_table(sens, pct_cols=["break_even_pd", "approval_rate", "bad_rate_approved"]),
        nl + "## Full strategy table (valid)" + nl,
        md_table(strategy, pct_cols=pct),
    ]
    report = resolve_path(cfg["reports"]["cutoff"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"Break-even PD {econ.break_even_pd:.4f}; PD-rule cutoff {cut}{nl}")
    print(points.round(4).to_string(), nl)
    print(bands.round(4).to_string(), nl)
    print(sens.round(4).to_string(), nl)
    print(f"Value over approve-all: ${value:,.0f} per {PER_N:,} (Illustrative). Report: {report}")


if __name__ == "__main__":
    main()
