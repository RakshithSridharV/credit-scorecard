"""Adverse-action reason codes from the points scorecard (D-019).

US law (ECOA / Regulation B) requires a lender that declines an applicant to
state the specific principal reasons. With a points scorecard this is a lookup:

    points lost on feature f = reference points for f - points the applicant got

where the reference is the AVERAGE points for f on train. The features with the
most points lost are the reasons: at most four, each costing at least
`min_points_lost`, and never none for a declined applicant. Because the same
table produces the score, the reasons explain the actual decision; nothing is
estimated.

Why the average and not the best bin: measured against the best bin, every
applicant under 63 "loses" age points for not being 68+, and age appeared in
the top four for 84% of declines. Against the average, a reason means "this
pulled you below a typical applicant", which is what a principal reason is.

    python -m scorecard.explain.reason_codes

computes the reference on train, saves it, and reports on valid declines. No test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import TARGET
from scorecard.data.split import load_split
from scorecard.evaluation.cutoff import economics_from_config, policy_cutoff
from scorecard.features.binning import assign_bins, bin_labels, missing_index
from scorecard.models.scorecard import Scorecard

ELDERLY_AGE = 62  # Regulation B 12 CFR 1002.6(b)(2): applicants 62+ must not get a negative value
REFERENCE_VERSION = 1


@dataclass(frozen=True)
class Reason:
    code: str
    text: str  # what the applicant reads


# Per feature: (reason when in a low-points numeric bin, reason when the value is missing).
# Bins are monotone (D-014), so losing points on a numeric bin always means
# "on the risky side", and one sentence per feature is enough.
# The three past-due features share one Missing reason: the 96/98 codes (D-006)
# make all three missing together, and the applicant should read it once.
_PAY_UNKNOWN = Reason("PAY_UNKNOWN", "Payment history could not be verified")
REASONS: dict[str, tuple[Reason, Reason | None]] = {
    "RevolvingUtilizationOfUnsecuredLines": (
        Reason("UTIL_HIGH", "Balances on revolving accounts are high relative to credit limits"),
        Reason("UTIL_UNKNOWN", "Revolving account utilisation could not be determined"),
    ),
    "NumberOfTimes90DaysLate": (
        Reason("DLQ_90", "Serious delinquency: payment 90 or more days past due"),
        _PAY_UNKNOWN,
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        Reason("DLQ_30", "Payments 30-59 days past due in the last two years"),
        _PAY_UNKNOWN,
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        Reason("DLQ_60", "Payments 60-89 days past due in the last two years"),
        _PAY_UNKNOWN,
    ),
    "age": (Reason("AGE", "Age of applicant"), None),
    "DebtRatio": (
        Reason("DEBT_HIGH", "Debt payments are high relative to income"),
        Reason("DEBT_UNKNOWN", "Debt burden could not be assessed: income not reported"),
    ),
    "MonthlyIncome": (
        Reason("INCOME_LOW", "Reported income is low"),
        Reason("INCOME_UNKNOWN", "Income not reported"),
    ),
    "NumberOfOpenCreditLinesAndLoans": (Reason("FEW_ACCOUNTS", "Few open credit accounts"), None),
    "NumberRealEstateLoansOrLines": (
        Reason("NO_REAL_ESTATE", "No real estate loans or lines on record"),
        None,
    ),
    "NumberOfDependents": (Reason("DEPENDENTS", "Number of dependents"), None),
}


def code_texts() -> dict[str, str]:
    """Every code -> applicant-facing text."""
    return {r.code: r.text for pair in REASONS.values() for r in pair if r is not None}


def reason_for(feature: str, is_missing: bool) -> Reason:
    numeric, if_missing = REASONS[feature]
    return (if_missing or numeric) if is_missing else numeric


# --------------------------------------------------------------------------- #
# Points per applicant and the reference they are compared with
# --------------------------------------------------------------------------- #
def applicant_points(sc: Scorecard, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Points received per feature (rows x features) and a matching is-missing mask."""
    pts, missing = {}, {}
    for f in sc.features:
        edges = sc.encoder_.features_[f].edges
        idx = assign_bins(X[f].to_numpy(dtype=float), edges)
        pts[f] = np.asarray(sc.points_[f])[idx]
        missing[f] = idx == missing_index(edges)
    return pd.DataFrame(pts, index=X.index), pd.DataFrame(missing, index=X.index)


def reference_points(sc: Scorecard, train: pd.DataFrame) -> dict[str, float]:
    """Average points per feature on train: the 'typical applicant' (learned, train only)."""
    pts, _ = applicant_points(sc, train)
    return {f: float(v) for f, v in pts.mean().items()}


def best_points(sc: Scorecard) -> dict[str, float]:
    """Most points available per feature, over bins that had train rows.
    Used only to compare against the rejected best-bin method in the report."""
    out = {}
    for f in sc.features:
        fb = sc.encoder_.features_[f]
        seen = [p for p, g, b in zip(sc.points_[f], fb.goods, fb.bads, strict=True) if g + b > 0]
        out[f] = float(max(seen))
    return out


def save_reference(reference: dict[str, float], path: Path | str) -> None:
    payload = {
        "schema_version": REFERENCE_VERSION,
        "method": "train_mean_points",
        "reference": reference,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_reference(path: Path | str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != REFERENCE_VERSION:
        raise ValueError(f"Unsupported reason reference version {payload.get('schema_version')}")
    return {f: float(v) for f, v in payload["reference"].items()}


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #
def top_reasons(
    sc: Scorecard,
    X: pd.DataFrame,
    reference: dict[str, float],
    max_reasons: int = 4,
    min_points_lost: float = 5.0,
) -> pd.DataFrame:
    """Up to `max_reasons` reason codes per row, most points lost first.

    A feature counts if it lost at least `min_points_lost`; if none does, the
    single largest loss is still given, so a declined applicant always gets a
    reason. Ties keep the scorecard's feature order (strongest first), so the
    output is deterministic. A code already given (PAY_UNKNOWN) is not repeated.
    """
    pts, missing = applicant_points(sc, X)
    lost = pd.Series(reference)[pts.columns].to_numpy() - pts.to_numpy()
    M, feats = missing.to_numpy(), list(pts.columns)
    rows = []
    for i in range(len(lost)):
        order = np.argsort(-lost[i], kind="stable")  # stable: ties keep feature order
        codes: list[str] = []
        sizes: list[float] = []
        for rank, j in enumerate(order):
            first_and_positive = rank == 0 and lost[i, j] > 0
            if len(codes) == max_reasons or not (
                lost[i, j] >= min_points_lost or first_and_positive
            ):
                break
            code = reason_for(feats[j], bool(M[i, j])).code
            if code not in codes:
                codes.append(code)
                sizes.append(round(float(lost[i, j]), 1))
        codes += [""] * (max_reasons - len(codes))
        sizes += [0.0] * (max_reasons - len(sizes))
        rows.append([*codes, *sizes])
    cols = [f"reason_{k}" for k in range(1, max_reasons + 1)]
    cols += [f"points_lost_{k}" for k in range(1, max_reasons + 1)]
    return pd.DataFrame(rows, index=X.index, columns=cols)


def reason_frequency(reasons: pd.DataFrame, max_reasons: int) -> pd.DataFrame:
    """Share of rows with each code as reason 1, and anywhere in the list."""
    cols = [f"reason_{k}" for k in range(1, max_reasons + 1)]
    texts = code_texts()
    first = reasons["reason_1"].value_counts(normalize=True)
    anywhere = pd.Series({c: (reasons[cols] == c).any(axis=1).mean() for c in texts})
    t = pd.DataFrame({"share_as_reason_1": first, f"share_in_top_{max_reasons}": anywhere})
    t = t.fillna(0).drop(index="", errors="ignore")
    t = t[t.sum(axis=1) > 0].sort_values(list(t.columns), ascending=False)
    t.insert(0, "text", [texts[c] for c in t.index])
    return t.rename_axis("code")


def elderly_rule_check(sc: Scorecard, ages: range = range(18, 111)) -> dict[str, int | bool]:
    """Regulation B: age may be used in an empirically derived scoring system only
    if applicants 62+ get no value less favourable than the best-treated under-62
    group. Check every whole age against the fitted points."""
    edges = sc.encoder_.features_["age"].edges
    pts = np.asarray(sc.points_["age"])[assign_bins(np.array(ages, dtype=float), edges)]
    a = np.array(ages)
    lowest_62_plus = int(pts[a >= ELDERLY_AGE].min())
    best_under_62 = int(pts[a < ELDERLY_AGE].max())
    return {
        "lowest_points_62_plus": lowest_62_plus,
        "best_points_under_62": best_under_62,
        "passes": lowest_62_plus >= best_under_62,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    rc = cfg["reasons"]
    n_max, min_lost = rc["max_reasons"], rc["min_points_lost"]
    sc = Scorecard.from_json(resolve_path(cfg["artifacts"]["scorecard"]))
    cut = policy_cutoff(sc, economics_from_config(cfg))
    interim = resolve_path(cfg["data"]["interim_dir"])
    train, valid = load_split("train", interim), load_split("valid", interim)

    reference = reference_points(sc, train)
    save_reference(reference, resolve_path(cfg["artifacts"]["reason_reference"]))

    scores = pd.Series(sc.score(valid), index=valid.index)
    declined = valid[scores < cut]
    reasons = top_reasons(sc, declined, reference, n_max, min_lost)
    texts = code_texts()
    freq = reason_frequency(reasons, n_max)

    # The rejected method, kept in the report as evidence for D-019.
    best_based = top_reasons(sc, declined, best_points(sc), n_max, min_points_lost=0)
    compare = pd.DataFrame(
        {
            "average, >= min (chosen): reason_1": freq["share_as_reason_1"],
            f"average, >= min (chosen): in top {n_max}": freq[f"share_in_top_{n_max}"],
            "best bin, > 0: reason_1": reason_frequency(best_based, n_max)["share_as_reason_1"],
            f"best bin, > 0: in top {n_max}": reason_frequency(best_based, n_max)[
                f"share_in_top_{n_max}"
            ],
        }
    ).fillna(0)

    n_given = (reasons[[f"reason_{k}" for k in range(1, n_max + 1)]] != "").sum(axis=1)
    count_dist = n_given.value_counts(normalize=True).sort_index().rename("share").to_frame()
    count_dist = count_dist.rename_axis("reasons given")

    bad_rate = (
        valid.loc[declined.index, TARGET]
        .groupby(reasons["reason_1"])
        .agg(["size", "mean"])
        .rename(columns={"size": "declined", "mean": "bad_rate"})
        .sort_values("declined", ascending=False)
    )

    reason_table = pd.DataFrame(
        [
            {"feature": f, "when": w, "code": r.code, "text": r.text}
            for f in sc.features
            for w, r in zip(("value in a low-points bin", "value missing"), REASONS[f], strict=True)
            if r is not None
        ]
    )
    ref_table = pd.DataFrame(
        {"reference (train mean)": reference, "best bin": best_points(sc)}
    ).rename_axis("feature")

    # Worked examples: three declined applicants, chosen by a fixed seed.
    pts, _ = applicant_points(sc, declined)
    examples = []
    for rid in declined.sample(3, random_state=cfg["split"]["seed"]).index:
        rows = []
        for f in sc.features:
            fb = sc.encoder_.features_[f]
            v = declined.at[rid, f]
            i = int(assign_bins([v], fb.edges)[0])
            rows.append(
                {
                    "feature": f,
                    "value": v,
                    "bin": bin_labels(fb.edges)[i],
                    "points": int(pts.at[rid, f]),
                    "reference": round(reference[f], 1),
                    "lost": round(reference[f] - pts.at[rid, f], 1),
                }
            )
        r = reasons.loc[rid]
        listed = [
            f"{k}. `{r[f'reason_{k}']}` {texts[r[f'reason_{k}']]} (-{r[f'points_lost_{k}']:g})"
            for k in range(1, n_max + 1)
            if r[f"reason_{k}"]
        ]
        examples.append((rid, int(scores[rid]), pd.DataFrame(rows), listed))

    age_check = elderly_rule_check(sc)

    nl = "\n"
    sections = [
        "# Reason codes" + nl,
        "Train-fitted scorecard and reference points; applicants are **valid** rows declined at "
        f"the D-018 cutoff ({cut}). Test is not used. "
        "Generated by `python -m scorecard.explain.reason_codes`." + nl,
        "Method (D-019): points lost = average points for the feature on train - points received. "
        f"A reason must cost >= {min_lost:g} points (**Assumption**: a quarter of the PDO); "
        f"at most {n_max}; the largest loss is always given." + nl,
        "## Reason code table" + nl,
        md_table(reason_table, index=False),
        nl + "## Reference points" + nl,
        md_table(ref_table),
        nl + f"## Declined applicants on valid: {len(declined):,} "
        f"({len(declined) / len(valid):.1%})" + nl,
        "### How often each reason appears" + nl,
        md_table(freq, pct_cols=["share_as_reason_1", f"share_in_top_{n_max}"]),
        nl + "### Chosen method vs best-bin method" + nl,
        md_table(compare, pct_cols=list(compare.columns)),
        nl + "### Bad rate by first reason" + nl,
        md_table(bad_rate, pct_cols=["bad_rate"]),
        nl + "### Number of reasons given" + nl,
        md_table(count_dist, pct_cols=["share"]),
        nl + "## Worked examples" + nl,
    ]
    for rid, s, table, listed in examples:
        sections += [
            f"### Row {rid}: score {s} (cutoff {cut})" + nl,
            md_table(table, index=False),
            nl + "Reasons given:" + nl,
            *listed,
            "",
        ]
    sections += [
        "## Regulation B age check" + nl,
        f"Lowest age points for applicants {ELDERLY_AGE}+: {age_check['lowest_points_62_plus']}. "
        f"Best age points under {ELDERLY_AGE}: {age_check['best_points_under_62']}. "
        f"**{'Pass' if age_check['passes'] else 'FAIL'}**: no applicant {ELDERLY_AGE}+ "
        "is scored below the best-treated younger group." + nl,
    ]
    report = resolve_path(cfg["reports"]["reason_codes"])
    report.write_text(nl.join(sections) + nl, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"Cutoff {cut}; declined {len(declined):,} of {len(valid):,}{nl}")
    print(ref_table.round(1).to_string(), nl)
    print(freq.round(4).to_string(), nl)
    print(compare.round(3).to_string(), nl)
    print(bad_rate.round(4).to_string(), nl)
    print(count_dist.round(4).to_string(), nl)
    for rid, s, table, listed in examples:
        print(f"Row {rid}, score {s}{nl}{table.to_string(index=False)}")
        print(*listed, sep=nl)
        print()
    print(age_check, nl, f"Report: {report}")


if __name__ == "__main__":
    main()
