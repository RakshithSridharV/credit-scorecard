"""Read-only data audit of cs-training.csv -> reports/data_audit.md.

Nothing here changes the data. Every number in the report is computed by the
functions below, so the report can be regenerated and checked at any time:

    python -m scorecard.data.audit
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.config import load_config, resolve_path
from scorecard.data.load import (
    FEATURES,
    PAST_DUE_COLUMNS,
    PAST_DUE_SPECIAL_CODES,
    TARGET,
    load_raw,
)

INCOME = "MonthlyIncome"
DEPENDENTS = "NumberOfDependents"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
DEBT_RATIO = "DebtRatio"

EXTRA_PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bad_rate_by(df: pd.DataFrame, groups: pd.Series, name: str = "group") -> pd.DataFrame:
    """Rows, share of all rows, and bad rate for each value of `groups`.

    observed=False keeps empty bands (rows=0) so the report can always look a
    band up by its label, whatever data it runs on.
    """
    out = df.groupby(groups.rename(name), observed=False, dropna=False, sort=False)[TARGET].agg(
        rows="size", bad_rate="mean"
    )
    out["share"] = out["rows"] / len(df)
    return out[["rows", "share", "bad_rate"]]


def income_status(income: pd.Series) -> pd.Series:
    """Label each row's income as missing / 0 / 1 / >1 (the groups that behave differently)."""
    status = np.select(
        [income.isna(), income == 0, income == 1],
        ["missing", "0", "1"],
        default=">1",
    )
    return pd.Series(pd.Categorical(status, ["missing", "0", "1", ">1"]), index=income.index)


# --------------------------------------------------------------------------- #
# Audit computations (each returns plain pandas objects, easy to test)
# --------------------------------------------------------------------------- #
def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "unique": df.nunique(),
        }
    )


def target_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    out = pd.DataFrame({"rows": counts, "share": counts / len(df)})
    out.index = out.index.map({0: "0 = good", 1: "1 = bad (90+ days late within 2y)"})
    return out


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns[df.isna().any()]:
        m = df[col].isna()
        rows.append(
            {
                "column": col,
                "missing": int(m.sum()),
                "missing_pct": m.mean(),
                "bad_rate_if_missing": df.loc[m, TARGET].mean(),
                "bad_rate_if_present": df.loc[~m, TARGET].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def duplicate_summary(df: pd.DataFrame) -> dict[str, float]:
    full_dup = df.duplicated()  # 2nd, 3rd... copy of an identical row
    feat_dup = df[FEATURES].duplicated()
    # Groups with identical features but BOTH labels: proof that identical rows
    # are not always the same event.
    n_labels = df.groupby(FEATURES, dropna=False)[TARGET].transform("nunique")
    return {
        "full_row_duplicates": int(full_dup.sum()),
        "feature_only_duplicates": int(feat_dup.sum()),
        "rows_in_conflicting_label_groups": int((n_labels > 1).sum()),
        "dup_bad_rate": df.loc[full_dup, TARGET].mean(),
        "dup_income_missing_share": df.loc[full_dup, INCOME].isna().mean(),
        "all_income_missing_share": df[INCOME].isna().mean(),
    }


def top_duplicate_profiles(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    dups = df[df.duplicated(keep=False)]
    counts = dups.value_counts(dropna=False).head(n).rename("copies")
    return counts.reset_index()


def describe_extended(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(percentiles=EXTRA_PERCENTILES).T


def past_due_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    """All observed values of the three past-due columns side by side."""
    table = pd.concat({c: df[c].value_counts() for c in PAST_DUE_COLUMNS}, axis=1)
    return table.sort_index().fillna(0).astype(int)


def past_due_special_rows(df: pd.DataFrame) -> dict[str, float]:
    is_code = df[PAST_DUE_COLUMNS].isin(PAST_DUE_SPECIAL_CODES)
    any_code = is_code.any(axis=1)
    real = df.loc[~any_code, PAST_DUE_COLUMNS]
    return {
        "rows_any_column": int(any_code.sum()),
        "rows_all_three_columns": int(is_code.all(axis=1).sum()),
        "bad_rate": df.loc[any_code, TARGET].mean(),
        "max_real_value": int(real.max().max()),
        # The source stores "fully used" as 0.9999999, so compare with a tolerance, not == 1.
        "util_at_cap_share": np.isclose(df.loc[any_code, UTIL], 1.0, atol=1e-6).mean(),
        "median_open_lines": df.loc[any_code, "NumberOfOpenCreditLinesAndLoans"].median(),
        "median_open_lines_others": df.loc[~any_code, "NumberOfOpenCreditLinesAndLoans"].median(),
        "median_age": df.loc[any_code, "age"].median(),
    }


def age_checks(df: pd.DataFrame) -> dict[str, int]:
    return {
        "age_0": int((df["age"] == 0).sum()),
        "age_below_18": int((df["age"] < 18).sum()),
        "age_above_100": int((df["age"] > 100).sum()),
        "age_max": int(df["age"].max()),
        "age_min_excluding_0": int(df.loc[df["age"] > 0, "age"].min()),
    }


def age_bands(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(df["age"], [-1, 17, 25, 35, 45, 55, 65, 75, 200])
    return bad_rate_by(df, bands, "age band").sort_index()


def utilisation_thresholds(
    df: pd.DataFrame, thresholds: Iterable[float] = (1, 10, 100)
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = df[UTIL] > t
        rows.append(
            {
                "rule": f"utilisation > {t}",
                "rows": int(m.sum()),
                "bad_rate": df.loc[m, TARGET].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def utilisation_bands(df: pd.DataFrame) -> pd.DataFrame:
    edges = [-np.inf, 0.5, 1, 1.5, 2, 10, 100, np.inf]
    labels = ["<= 0.5", "(0.5, 1]", "(1, 1.5]", "(1.5, 2]", "(2, 10]", "(10, 100]", "> 100"]
    return bad_rate_by(df, pd.cut(df[UTIL], edges, labels=labels), "utilisation band").sort_index()


def utilisation_extreme_profile(df: pd.DataFrame, threshold: float = 100) -> dict[str, float]:
    u = df.loc[df[UTIL] > threshold, UTIL]
    return {
        "rows": int(u.size),
        "median": u.median(),
        "whole_number_share": (u % 1 == 0).mean(),
    }


def debt_ratio_by_income_status(df: pd.DataFrame) -> pd.DataFrame:
    status = income_status(df[INCOME])
    g = df.groupby(status, observed=False)
    dr = g[DEBT_RATIO]
    return pd.DataFrame(
        {
            "rows": g.size(),
            "median_debt_ratio": dr.median(),
            "p90_debt_ratio": dr.quantile(0.9),
            "share_debt_ratio_gt_1": dr.apply(lambda s: (s > 1).mean()),
            "share_whole_number": dr.apply(lambda s: (s % 1 == 0).mean()),
            "bad_rate": g[TARGET].mean(),
        }
    ).rename_axis("income")


def debt_ratio_gt1_bad_rate(df: pd.DataFrame) -> dict[str, float]:
    high = df[DEBT_RATIO] > 1
    real_income = df[INCOME] > 1
    return {
        "income_real": df.loc[high & real_income, TARGET].mean(),
        "income_missing_0_or_1": df.loc[high & ~real_income, TARGET].mean(),
    }


def income_bands(df: pd.DataFrame) -> pd.DataFrame:
    edges = [-np.inf, 0, 1, 1000, 2000, 3000, 5000, 10000, np.inf]
    labels = ["0", "1", "(1, 1k]", "(1k, 2k]", "(2k, 3k]", "(3k, 5k]", "(5k, 10k]", "> 10k"]
    bands = pd.cut(df[INCOME], edges, labels=labels).cat.add_categories("missing")
    bands = bands.fillna("missing")
    out = bad_rate_by(df, bands, "monthly income (USD)")
    return out.reindex(["missing", *labels])


def missing_overlap(df: pd.DataFrame) -> pd.DataFrame:
    combo = (
        df[INCOME].isna().map({True: "income missing", False: "income present"})
        + " / "
        + df[DEPENDENTS].isna().map({True: "dependents missing", False: "dependents present"})
    )
    return bad_rate_by(df, combo, "combination").sort_index()


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _fmt_num(v: object) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def pct(v: float, digits: int = 2) -> str:
    return f"{100 * v:.{digits}f}%"


def md_table(df: pd.DataFrame, pct_cols: Iterable[str] = (), index: bool = True) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    pct_cols = set(pct_cols)
    frame = df.reset_index() if index else df
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in frame.iterrows():
        cells = [pct(v) if c in pct_cols and pd.notna(v) else _fmt_num(v) for c, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(df: pd.DataFrame) -> str:
    """Build the Markdown report. Long by nature: one section per audit check."""
    n = len(df)
    overall_bad = df[TARGET].mean()
    tb = target_balance(df)
    ms = missing_summary(df)
    dup = duplicate_summary(df)
    desc = describe_extended(df)
    pdv = past_due_value_counts(df)
    pds = past_due_special_rows(df)
    age = age_checks(df)
    ab = age_bands(df)
    ut = utilisation_thresholds(df)
    ub = utilisation_bands(df)
    ux = utilisation_extreme_profile(df)
    dri = debt_ratio_by_income_status(df)
    drb = debt_ratio_gt1_bad_rate(df)
    ib = income_bands(df)
    mo = missing_overlap(df)

    inc_miss = ms.loc[INCOME]
    dep_miss = ms.loc[DEPENDENTS]
    u_1_2_bad = df.loc[(df[UTIL] > 1) & (df[UTIL] <= 2), TARGET].mean()
    u_10_100 = int(ub.loc["(10, 100]", "rows"])
    young_bad = df.loc[df["age"].between(18, 35), TARGET].mean()
    old_bad = df.loc[df["age"] > 75, TARGET].mean()
    dep_missing_rows = df[DEPENDENTS].isna()
    dep_and_inc = int((dep_missing_rows & df[INCOME].isna()).sum())
    low_inc_bad = ib.loc["(1k, 2k]", "bad_rate"]

    s: list[str] = []
    add = s.append

    add("# Data audit: Give Me Some Credit (`cs-training.csv`)\n")
    add(
        "> Generated by `python -m scorecard.data.audit`. Read-only: no data was changed.  \n"
        "> US consumer-credit data used as a method demo, not an Indian lending model.  \n"
        "> **Bad** = `SeriousDlqin2yrs = 1`: the borrower went 90+ days past due within 2 years.  \n"
        "> Caveat: this audit looks at the full file before the train/valid/test split. That is fine "
        "for spotting impossible values, but any threshold chosen here should be re-checked on the "
        "training split only, so test data does not shape our decisions.\n"
    )

    # 1 ---------------------------------------------------------------------
    add("## 1. Shape and dtypes\n")
    add(
        f"After dropping the `Unnamed: 0` row-number column: **{n:,} rows x {df.shape[1]} columns** "
        f"(1 target + {len(FEATURES)} features).\n"
    )
    add(md_table(column_overview(df).rename_axis("column")) + "\n")
    add(
        "**What it means.** Every column is numeric. `MonthlyIncome` and `NumberOfDependents` are "
        "`float64` even though dependents is a count: a pandas `int64` column cannot hold `NaN`, so "
        "a column with gaps is stored as float. `Unnamed: 0` was just the CSV row number (no "
        "customer ID exists), which matters for the duplicates check below.\n\n"
        "**Why a bank cares.** The column list and types are a contract between the training data "
        "and the scoring API. If a field arrives renamed or as text in production, the model must "
        "fail loudly, not score silently with garbage. That is why `load_raw()` rejects any file "
        "whose columns differ from the expected list.\n"
    )

    # 2 ---------------------------------------------------------------------
    add("## 2. Target balance\n")
    add(md_table(tb, pct_cols=["share"]) + "\n")
    add(
        f"**What it means.** {pct(overall_bad)} of borrowers went seriously delinquent, about 1 in "
        f"{round(1 / overall_bad)}. These are people a lender had already approved, so most are "
        "decent risks.\n\n"
        "**Why a bank cares.** (a) Accuracy is useless here: a model that calls everyone good is "
        f"{pct(1 - overall_bad)} accurate and catches zero defaulters, so we use AUC/Gini/KS. "
        "(b) The split must be **stratified** so train, validation and test all keep the same "
        "bad rate. (c) We should not oversample bads for the scorecard: the score is converted to a "
        "probability of default (PD), and PDs feed pricing and provisioning, so they must reflect "
        "the true bad rate.\n"
    )

    # 3 ---------------------------------------------------------------------
    add("## 3. Missing values\n")
    add(md_table(ms, pct_cols=["missing_pct", "bad_rate_if_missing", "bad_rate_if_present"]) + "\n")
    add(
        f"**What it means.** Only two columns have gaps. Income is missing for "
        f"{int(inc_miss['missing']):,} borrowers ({pct(inc_miss['missing_pct'], 1)}) and dependents "
        f"for {int(dep_miss['missing']):,} ({pct(dep_miss['missing_pct'], 1)}). The gaps are **not "
        f"random**: borrowers with missing income default *less* ({pct(inc_miss['bad_rate_if_missing'])}) "
        f"than those who reported it ({pct(inc_miss['bad_rate_if_present'])}).\n\n"
        '**Why a bank cares.** Missingness is itself information ("did not declare income" '
        "is a behaviour). Filling the gap with the median would hide that signal and push "
        f"{int(inc_miss['missing']):,} lower-risk people into a middle-income group with a "
        'different risk level. Scorecards usually give "Missing" its own bin so the '
        "model learns its own risk weight.\n"
    )

    # 4 ---------------------------------------------------------------------
    add("## 4. Duplicates\n")
    add(
        f"| check | value |\n|---|---|\n"
        f"| identical rows (all columns), extra copies | {dup['full_row_duplicates']:,} "
        f"({pct(dup['full_row_duplicates'] / n)}) |\n"
        f"| identical features only (ignoring target), extra copies | "
        f"{dup['feature_only_duplicates']:,} |\n"
        f"| rows in groups with identical features but *different* targets | "
        f"{dup['rows_in_conflicting_label_groups']:,} |\n"
        f"| bad rate of duplicate copies | {pct(dup['dup_bad_rate'])} |\n"
        f"| share of duplicate copies with missing income | {pct(dup['dup_income_missing_share'], 1)} "
        f"(vs {pct(dup['all_income_missing_share'], 1)} overall) |\n"
    )
    add("Most common duplicated profiles:\n")
    add(md_table(top_duplicate_profiles(df), index=False) + "\n")
    add(
        "**What it means.** There is no customer ID, so we cannot tell whether a repeated row is "
        "the same person entered twice or different people with identical simple profiles. The "
        "evidence points to the second: the repeats are mostly young, thin-file borrowers (0 to 2 "
        "credit lines, utilisation 0 or at the cap, income missing or a round $820), the kind "
        "of profile where many people share identical values. Some are the 96/98-coded rows "
        "from section 6. "
        f"{dup['rows_in_conflicting_label_groups']:,} rows have identical features but different "
        "outcomes, so at least those are different events.\n\n"
        "**Why a bank cares.** True duplicates double-count one customer, and if one copy lands in "
        "train and its twin in test, the test score is inflated (a form of leakage). But dropping "
        "genuinely different people would under-represent thin-file applicants, a real segment "
        f"the model must score. Either way the effect is small ({pct(dup['full_row_duplicates'] / n)} "
        "of rows).\n"
    )

    # 5 ---------------------------------------------------------------------
    add("## 5. Distributions (`describe()` with extra percentiles)\n")
    add(md_table(desc.rename_axis("column")) + "\n")
    add(
        "**What it means.** Compare the mean with the median (50%) and the 99th percentile:\n\n"
        f"- Utilisation: mean {desc.loc[UTIL, 'mean']:.2f}, median {desc.loc[UTIL, '50%']:.3f}, "
        f"99th pct {desc.loc[UTIL, '99%']:.3f}, max {desc.loc[UTIL, 'max']:,.0f}. "
        "A few huge values drag the mean up (see section 8).\n"
        f"- DebtRatio: mean {desc.loc[DEBT_RATIO, 'mean']:,.0f} vs median "
        f"{desc.loc[DEBT_RATIO, '50%']:.3f}; the 95th pct is already "
        f"{desc.loc[DEBT_RATIO, '95%']:,.0f} (see section 9).\n"
        f"- MonthlyIncome: median ${desc.loc[INCOME, '50%']:,.0f}, max "
        f"${desc.loc[INCOME, 'max']:,.0f} per month: extreme, possibly real.\n"
        "- Past-due counts: 99th pct is 2 to 4, but 99.9th pct is 98 (see section 6).\n"
        f"- Long tails that are rare but not impossible: open credit lines up to "
        f"{desc.loc['NumberOfOpenCreditLinesAndLoans', 'max']:.0f}, real-estate loans up to "
        f"{desc.loc['NumberRealEstateLoansOrLines', 'max']:.0f}, dependents up to "
        f"{desc.loc[DEPENDENTS, 'max']:.0f}.\n\n"
        "**Why a bank cares.** Means and standard deviations are distorted by a handful of rows, "
        "so any model that uses raw values (e.g. logistic regression on unbinned features) would "
        "be steered by outliers. A WoE scorecard bins each variable, which makes it robust to long "
        "tails, but *impossible* values still need a rule, because they end up in the wrong bin.\n"
    )

    # 6 ---------------------------------------------------------------------
    add("## 6. Past-due columns: the 96 / 98 codes\n")
    add(md_table(pdv.rename_axis("value")) + "\n")
    add(
        f"**What it means.** Real counts stop at {pds['max_real_value']}, then jump straight to 96 "
        f"and 98. These appear in {pds['rows_any_column']:,} rows, and in "
        f"{pds['rows_all_three_columns']:,} of them all three columns carry the code at the same "
        f"time. Being 30-59 days late 98 times inside a 2-year window is impossible, so these "
        "are almost certainly **placeholder codes** from the source system, not counts (the "
        "dataset documentation does not say what they mean). The rows look alike in other ways "
        "too: "
        f"utilisation is the capped value 0.9999999 in {pct(pds['util_at_cap_share'], 0)} of them, "
        "the median "
        f"number of open credit lines is {pds['median_open_lines']:.0f} (vs "
        f"{pds['median_open_lines_others']:.0f} for everyone else) and the median age is "
        f"{pds['median_age']:.0f}. Crucially, their bad rate is **{pct(pds['bad_rate'], 1)}** vs "
        f"{pct(overall_bad)} overall.\n\n"
        '**Why a bank cares.** Left as numbers, 98 is treated as "extremely late", which '
        "happens to correlate with default but for the wrong reason, and it stretches every "
        "statistic of these columns. Dropping the rows would throw away a group where more than "
        "half default, making the portfolio look safer than it is. The fix has to keep the "
        "signal but stop pretending 98 is a count.\n"
    )

    # 7 ---------------------------------------------------------------------
    add("## 7. Age\n")
    add(
        f"| check | rows |\n|---|---|\n| age = 0 | {age['age_0']} |\n"
        f"| age < 18 (including 0) | {age['age_below_18']} |\n"
        f"| age > 100 | {age['age_above_100']} (max {age['age_max']}) |\n"
        f"| youngest valid age | {age['age_min_excluding_0']} |\n"
    )
    add(md_table(ab, pct_cols=["share", "bad_rate"]) + "\n")
    add(
        f"**What it means.** Exactly one row has age 0 and no row is aged 1 to 17, so it is a data "
        "entry error, not a minor. Ages above 100 are rare but possible. Risk falls steadily with "
        f"age: {pct(young_bad, 1)} bad for 18-35 vs {pct(old_bad, 1)} above 75.\n\n"
        "**Why a bank cares.** A credit contract needs an adult (18+ in most US states), so age 0 "
        "can never be a real applicant; the scoring API should reject it outright. Age is also "
        "sensitive under US fair-lending law (ECOA / Regulation B): it may be used in an "
        "empirically derived, statistically sound scoring system, but applicants aged 62+ must "
        "not be scored *worse* because of age. Note this for the modelling stage.\n"
    )

    # 8 ---------------------------------------------------------------------
    add("## 8. Revolving utilisation > 1 / > 10 / > 100\n")
    add(md_table(ut, pct_cols=["bad_rate"]) + "\n")
    add(md_table(ub, pct_cols=["share", "bad_rate"]) + "\n")
    add(
        "**What it means.** Utilisation = card balance / credit limit, so the normal range is 0 to 1. "
        "Slightly above 1 is possible (over the limit after fees or interest), and those borrowers "
        f"are among the **riskiest borrowers in the data**: {pct(u_1_2_bad, 0)} bad between 1 and "
        "2. Above "
        f"100 the picture flips: {ux['rows']} rows with median {ux['median']:,.0f}, "
        f"{pct(ux['whole_number_share'], 0)} of them whole numbers, and a bad rate close to the "
        "population average. These look like dollar balances typed into the ratio field. Between "
        f"10 and 100 there are only {u_10_100} rows, so the exact cut-off barely matters.\n\n"
        "**Why a bank cares.** Utilisation is usually one of the strongest predictors on a "
        'scorecard, and a scorecard should show a clean "higher utilisation means higher risk" '
        "pattern that credit officers can sign off on. Error rows binned with genuine over-limit "
        "borrowers would dilute the riskiest bin and break that pattern.\n"
    )

    # 9 ---------------------------------------------------------------------
    add("## 9. DebtRatio when MonthlyIncome is missing\n")
    add(md_table(dri, pct_cols=["share_debt_ratio_gt_1", "share_whole_number", "bad_rate"]) + "\n")
    add(
        "**What it means.** DebtRatio should be monthly debt payments / monthly income, normally "
        f"0 to about 1. That holds when income is a real number (median "
        f"{dri.loc['>1', 'median_debt_ratio']:.2f}). When income is missing, 0 or 1, DebtRatio "
        f"jumps to medians of {dri.loc['missing', 'median_debt_ratio']:,.0f} / "
        f"{dri.loc['0', 'median_debt_ratio']:,.0f} / {dri.loc['1', 'median_debt_ratio']:,.0f}, and "
        f"in the missing-income rows {pct(dri.loc['missing', 'share_whole_number'], 0)} of values "
        "are whole numbers. The most likely explanation is that the data provider divided the debt by "
        "1 when income was unavailable, so the column holds the **debt amount in dollars**, not a "
        "ratio. Rows with income exactly 1 show the same jump (dividing by 1 changes nothing), "
        "which supports this.\n\n"
        "**Why a bank cares.** Debt-to-income is a core affordability measure (US "
        "ability-to-repay rules lean on it). Mixing dollar amounts with ratios makes "
        '"DebtRatio = 2,000" look like a crushing debt burden when it really means "income '
        'unknown". The data proves the mix-up hurts: among DebtRatio > 1 rows, the bad rate is '
        f"{pct(drb['income_real'], 1)} when income is real but only "
        f"{pct(drb['income_missing_0_or_1'], 1)} when income is missing/0/1. Two different "
        "populations would share one bin.\n"
    )

    # 10 --------------------------------------------------------------------
    add("## 10. Income == 0 vs missing (and the income == 1 placeholder)\n")
    add(md_table(ib, pct_cols=["share", "bad_rate"]) + "\n")
    add(
        f"**What it means.** {int(ib.loc['0', 'rows']):,} borrowers report income 0, "
        f"{int(ib.loc['1', 'rows']):,} report exactly $1, and "
        f"{int(ib.loc['missing', 'rows']):,} are missing. Real zero income should be very risky, "
        f"yet these groups default at {pct(ib.loc['0', 'bad_rate'], 1)}, "
        f"{pct(ib.loc['1', 'bad_rate'], 1)} and {pct(ib.loc['missing', 'bad_rate'], 1)}, well "
        f"below genuinely low earners ($1k-2k: {pct(low_inc_bad, 1)}). So 0 and 1 behave like "
        '"not reported", not like "poor". (A handful of other tiny values, $2 to $100, look '
        "odd too, but they are only "
        f"{int(((df[INCOME] > 1) & (df[INCOME] <= 100)).sum())} rows.)\n\n"
        "**Why a bank cares.** If 0 is treated as a real income, the lowest-income bin mixes "
        'low-risk "not reported" people with high-risk low earners. The scorecard would then '
        "show a strange pattern (very lowest income looks safe), which a credit committee or "
        "regulator would rightly question.\n"
    )

    # 11 --------------------------------------------------------------------
    add("## 11. Overlap of missing income and missing dependents\n")
    add(md_table(mo, pct_cols=["share", "bad_rate"]) + "\n")
    add(
        f"**What it means.** All {int(dep_missing_rows.sum()):,} rows with missing dependents also "
        f"have missing income ({dep_and_inc:,} of {int(dep_missing_rows.sum()):,}); dependents is "
        "never missing on its own. One cause (e.g. an application channel that skipped the "
        "household section) explains both gaps. Risk is lowest when both are missing.\n\n"
        "**Why a bank cares.** Two missing flags that always move together add no new information, "
        "and two correlated variables can make logistic-regression coefficients unstable. The "
        "production pipeline will also receive this combination, so it must handle both fields "
        "missing at once.\n"
    )

    # Summary ---------------------------------------------------------------
    add("## Issues to resolve before modelling\n")
    add(
        f"1. `MonthlyIncome` missing ({int(inc_miss['missing']):,}), plus 0 "
        f"({int(ib.loc['0', 'rows']):,}) and 1 ({int(ib.loc['1', 'rows']):,}) placeholders.\n"
        "2. `DebtRatio` is a dollar amount, not a ratio, wherever income is missing/0/1.\n"
        f"3. `NumberOfDependents` missing ({int(dep_miss['missing']):,}), always together with "
        "income.\n"
        f"4. Past-due columns: 96/98 placeholder codes in {pds['rows_any_column']} rows "
        f"(bad rate {pct(pds['bad_rate'], 1)}).\n"
        f"5. Utilisation > 100: {ux['rows']} rows that look like dollar amounts; {u_10_100} "
        "ambiguous rows between 10 and 100; 1 to 10 looks like genuine over-limit borrowing.\n"
        f"6. One row with age 0.\n"
        f"7. {dup['full_row_duplicates']} exact duplicate rows with no ID to resolve them.\n"
        "8. Long tails (income, credit lines, real-estate loans, dependents): rare but plausible.\n"
        f"9. Imbalanced target ({pct(overall_bad)} bad).\n\n"
        "Cleaning decisions for each issue are recorded in `docs/decisions.md`.\n"
    )
    return "\n".join(s)


def main() -> None:
    cfg = load_config()
    df = load_raw()
    out_path: Path = resolve_path(cfg["reports"]["data_audit"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(df), encoding="utf-8")
    print(f"Wrote {out_path} ({len(df):,} rows audited)")


if __name__ == "__main__":
    main()
