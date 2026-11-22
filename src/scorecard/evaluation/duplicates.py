"""Required D-009 check: AUC with and without duplicate rows, on valid and test (D-033).

D-009 kept the 609 exact duplicate rows (there is no customer ID, so identical
rows may be different people) and required this check at evaluation time. The
worry is leakage: if one copy of an applicant sits in train and its twin in
test, the test AUC is partly a memory test. Removing those rows and re-scoring
shows how much of the AUC depends on them.

"Without duplicates" (the D-009 definition): drop valid/test rows whose feature
values exactly match a train row, and keep one copy of rows repeated inside the
split. It is applied to two versions of the features:

* raw values, as they arrived (primary: the closest thing to "same person");
* cleaned values, the model's inputs (stricter: cleaning turns placeholders
  into NaN, so rows that differed only in a placeholder now collide).

This reads TEST outcomes, so it is NOT a training stage. It scores the frozen,
tested model only (fingerprints checked against the lock first), changes no
artifact, and records itself in the lock as an addendum, so the extra look at
test is on file.

    python -m scorecard.evaluation.duplicates           # once; recorded in the lock
    python -m scorecard.evaluation.duplicates --force   # re-run, recorded again
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scorecard.config import load_config, resolve_path
from scorecard.data.audit import md_table
from scorecard.data.load import FEATURES, TARGET
from scorecard.data.split import load_split
from scorecard.integrity import sha256
from scorecard.models.scorecard import Scorecard

CHECK = "D-009 duplicates"
SPLITS = ("valid", "test")


# --------------------------------------------------------------------------- #
# The definition
# --------------------------------------------------------------------------- #
def row_keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One 64-bit hash per row of `cols`. Unlike tuple comparison, NaN equals NaN
    here, so two rows both missing income count as identical. A collision among
    ~150,000 rows has probability about 1e-9: negligible."""
    return pd.util.hash_pandas_object(df[cols], index=False)


def keep_mask(train_keys: pd.Series, split_keys: pd.Series) -> pd.DataFrame:
    """Which split rows survive the D-009 definition, and why the others do not.

    Returns boolean columns `in_train` (an exact twin exists in train),
    `repeat_in_split` (2nd, 3rd... copy inside this split) and `keep`.
    """
    in_train = split_keys.isin(set(train_keys)).to_numpy()
    repeat = split_keys.duplicated(keep="first").to_numpy()
    return pd.DataFrame(
        {"in_train": in_train, "repeat_in_split": repeat, "keep": ~in_train & ~repeat},
        index=split_keys.index,
    )


def auc_with_without(y: np.ndarray, p: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    with_, without = roc_auc_score(y, p), roc_auc_score(y[keep], p[keep])
    return {
        "auc_all_rows": with_,
        "auc_without_duplicates": without,
        "auc_change": without - with_,
        "gini_change": 2 * (without - with_),
    }


# --------------------------------------------------------------------------- #
# Guard and record (the test lock, D-020)
# --------------------------------------------------------------------------- #
def verify_frozen(lock: Path, files: dict[str, Path]) -> None:
    """Refuse to run unless these files are byte-identical to the tested ones,
    so the check describes the model that was evaluated, not a later rebuild."""
    tested = json.loads(lock.read_text(encoding="utf-8"))["runs"][0]["fingerprints"]
    bad = [k for k, p in files.items() if sha256(p) != tested[k]]
    if bad:
        raise RuntimeError(f"{bad} differ from the tested version in {lock.name}")


def check_addendum(lock: Path, force: bool) -> None:
    state = json.loads(lock.read_text(encoding="utf-8"))
    done = [a for a in state.get("addenda", []) if a["check"] == CHECK]
    if done and not force:
        raise RuntimeError(
            f"{CHECK} was already run at {done[0]['at']} (see {lock}). "
            "Re-run with --force; the re-run will be recorded."
        )


def record_addendum(lock: Path, prints: dict[str, str], report: str) -> None:
    """Append to the lock's `addenda` list. `runs` is untouched: the API and the
    rebuild check read runs[0] only, so the official test run stays the reference."""
    state: dict[str, Any] = json.loads(lock.read_text(encoding="utf-8"))
    state.setdefault("addenda", []).append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "check": CHECK,
            "purpose": "sensitivity check required by D-009; nothing was tuned on test",
            "report": report,
            "fingerprints": prints,
        }
    )
    lock.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-run; recorded in the lock")
    args = parser.parse_args(argv)

    cfg = load_config()
    raw_dir = resolve_path(cfg["data"]["processed_dir"])  # raw splits
    clean_dir = resolve_path(cfg["data"]["interim_dir"])  # cleaned splits (model inputs)
    lock = resolve_path(cfg["evaluation"]["lock"])
    frozen = {
        "scorecard": resolve_path(cfg["artifacts"]["scorecard"]),
        "test_data": clean_dir / "test.csv",
    }
    verify_frozen(lock, frozen)  # before reading any outcome
    check_addendum(lock, args.force)

    sc = Scorecard.from_json(frozen["scorecard"])
    train = {"raw": load_split("train", raw_dir), "cleaned": load_split("train", clean_dir)}
    rows = []
    for split in SPLITS:
        data = {"raw": load_split(split, raw_dir), "cleaned": load_split(split, clean_dir)}
        scored = data["cleaned"]
        # Raw and cleaned files hold the same rows (cleaning never drops valid/test
        # rows), so a mask built on raw values can index the cleaned rows.
        if not data["raw"].index.equals(scored.index):
            raise RuntimeError(f"raw and cleaned {split} rows are not aligned")
        y = scored[TARGET].to_numpy()
        p = sc.score_pd(scored)  # the deployed PD, as in the headline (D-016)
        for version in ("raw", "cleaned"):
            m = keep_mask(row_keys(train[version], FEATURES), row_keys(data[version], FEATURES))
            rows.append(
                {
                    "split": split,
                    "matched on": f"{version} values",
                    "rows": len(scored),
                    "twin_in_train": int(m["in_train"].sum()),
                    "repeat_in_split": int(m["repeat_in_split"].sum()),
                    "removed": int((~m["keep"]).sum()),
                    **auc_with_without(y, p, m["keep"].to_numpy()),
                }
            )
    table = pd.DataFrame(rows).set_index(["split", "matched on"])
    largest = table["auc_change"].abs().max()
    material = cfg["duplicates"]["material_auc_change"]
    verdict = (
        f"below the {material} materiality threshold (**Assumption**, config `duplicates`), so "
        "the duplicates do not inflate the reported performance. D-009's choice to keep them "
        "stands."
        if largest < material
        else f"at or above the {material} materiality threshold (**Assumption**): the reported "
        "AUC partly depends on duplicates, and D-009 must be revisited."
    )

    nl = "\n"
    sections = [
        "# D-009 check: AUC with and without duplicate rows" + nl,
        "Required by D-009, run at the evaluation stage (D-033). The frozen, tested scorecard "
        "(fingerprints checked against the test lock) scores **valid** and **test**. Nothing is "
        "refitted or tuned. This is a recorded addendum in the test lock, not a second "
        "evaluation. Generated by `python -m scorecard.evaluation.duplicates`." + nl,
        "**Definition (D-009).** Drop valid/test rows whose 10 feature values exactly match a "
        "train row, and keep one copy of rows repeated inside the split. Applied to:" + nl,
        (
            "- **raw values** (primary): the data as it arrived, the closest proxy for the same "
            "applicant;\n"
            "- **cleaned values** (stricter): the model's inputs. Cleaning turns placeholders "
            "into missing, so rows that differed only in a placeholder now count as twins.\n"
        ),
        "Scored with the deployed PD (integer score -> master scale, D-016), as in the "
        "test-evaluation headline, so `auc_all_rows` matches that report." + nl,
        # Signed, fixed decimals: changes this small would otherwise print as -4e-05.
        md_table(
            table.round(4).assign(
                auc_change=table["auc_change"].map("{:+.5f}".format),
                gini_change=table["gini_change"].map("{:+.5f}".format),
            )
        ),
        nl + f"**Result.** The largest absolute AUC change is **{largest:.4f}**, {verdict} "
        "For scale: the test AUC's 95% bootstrap interval in "
        "[test_evaluation.md](test_evaluation.md) is about 0.02 wide." + nl,
    ]
    report = resolve_path(cfg["reports"]["duplicate_check"])
    report.write_text(nl.join(sections), encoding="utf-8", newline="\n")
    prints = {k: sha256(p) for k, p in frozen.items()}
    record_addendum(lock, prints, cfg["reports"]["duplicate_check"])  # only after the report

    pd.set_option("display.width", 220)
    print(table.round(5).to_string(), nl)
    print(f"Largest |AUC change|: {largest:.5f}{nl}Report: {report}")


if __name__ == "__main__":
    main()
