"""Rebuild every artifact from the raw CSV, in order, then prove it is the tested model (D-022).

    python -m scorecard.pipelines.train              # rebuild + verify against the test lock
    python -m scorecard.pipelines.train --no-verify  # rebuild only (e.g. a deliberate redevelopment)

Stages, each one a module that reads the config and the previous stage's files:

    audit -> split -> clean -> WoE encoder -> feature selection -> scorecard
    -> challenger -> cutoff report -> reason reference -> monitoring baseline

The test evaluation is NOT a stage: training must never look at test outcomes,
and test is scored once, separately (D-020). After the rebuild, the SHA-256 of
each artifact is compared with the fingerprints in the test lock: a match proves
that the model which was tested can be regenerated from raw data, code and config.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from scorecard.config import load_config, resolve_path
from scorecard.pipelines.evaluate import sha256

# (label, module whose main() runs the stage). Order matters: each reads the last one's output.
STAGES: list[tuple[str, str]] = [
    ("data audit", "scorecard.data.audit"),
    ("split", "scorecard.data.split"),
    ("clean", "scorecard.data.clean"),
    ("WoE encoder", "scorecard.features.encoder"),
    ("feature selection", "scorecard.features.selection"),
    ("scorecard", "scorecard.models.scorecard"),
    ("challenger", "scorecard.models.challenger"),
    ("cutoff report", "scorecard.evaluation.cutoff"),
    ("reason reference", "scorecard.explain.reason_codes"),
    ("monitoring baseline", "scorecard.evaluation.psi"),
]

# Lock key -> config key under `artifacts` (or the cleaned test file).
VERIFIED = {
    "scorecard": "scorecard",
    "challenger": "challenger",
    "reason_reference": "reason_reference",
}


def run_stages(stages: list[tuple[str, str]]) -> pd.DataFrame:
    """Import and run each stage's main(); return the time each took."""
    timings = []
    for i, (label, module) in enumerate(stages, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(stages)}] {label}  ({module})\n{'=' * 70}", flush=True)
        start = time.perf_counter()
        importlib.import_module(module).main()
        timings.append({"stage": label, "seconds": round(time.perf_counter() - start, 1)})
    return pd.DataFrame(timings).set_index("stage")


def compare_with_lock(lock_path: Path, current: dict[str, Path]) -> pd.DataFrame:
    """Rebuilt file hashes vs the FIRST (official) test-lock run."""
    tested = json.loads(lock_path.read_text(encoding="utf-8"))["runs"][0]["fingerprints"]
    rows = []
    for key, path in current.items():
        now = sha256(path)
        rows.append(
            {
                "artifact": key,
                "tested_sha256": tested[key][:16],
                "rebuilt_sha256": now[:16],
                "match": now == tested[key],
            }
        )
    return pd.DataFrame(rows).set_index("artifact")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild all artifacts from raw data.")
    parser.add_argument("--no-verify", action="store_true", help="skip the lock comparison")
    args = parser.parse_args(argv)

    timings = run_stages(STAGES)
    print(f"\n{timings.to_string()}\nTotal: {timings['seconds'].sum():.0f} s")

    if args.no_verify:
        return
    cfg = load_config()
    lock = resolve_path(cfg["evaluation"]["lock"])
    if not lock.exists():
        print("No test lock yet: nothing to verify against.")
        return
    current = {k: resolve_path(cfg["artifacts"][v]) for k, v in VERIFIED.items()}
    current["test_data"] = resolve_path(cfg["data"]["interim_dir"]) / "test.csv"
    table = compare_with_lock(lock, current)
    print(f"\nReproducibility vs the tested model ({lock.name}):\n{table.to_string()}")
    if not table["match"].all():
        print("\nFAIL: the rebuilt model is not the model that was tested.", file=sys.stderr)
        raise SystemExit(1)
    print("\nOK: rebuilt artifacts are byte-identical to the tested ones.")


if __name__ == "__main__":
    main()
