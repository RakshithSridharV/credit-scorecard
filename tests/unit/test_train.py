"""Training pipeline: stage order, no test evaluation inside, lock comparison."""

from __future__ import annotations

import importlib
import json

from scorecard.pipelines.evaluate import sha256
from scorecard.pipelines.train import STAGES, compare_with_lock, run_stages


def test_every_stage_has_a_main():
    for _, module in STAGES:
        assert callable(importlib.import_module(module).main), module


def test_stage_order_follows_the_data_flow():
    order = [m for _, m in STAGES]
    assert order.index("scorecard.data.split") < order.index("scorecard.data.clean")
    assert order.index("scorecard.data.clean") < order.index("scorecard.features.encoder")
    assert order.index("scorecard.features.selection") < order.index("scorecard.models.scorecard")
    # Everything downstream of the scorecard reads models/scorecard.json.
    for later in ("challenger", "cutoff", "reason_codes", "psi"):
        assert order.index("scorecard.models.scorecard") < next(
            i for i, m in enumerate(order) if m.endswith(later)
        )


def test_training_never_scores_test():
    assert "scorecard.pipelines.evaluate" not in [m for _, m in STAGES]


def test_run_stages_calls_main_in_order(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, name):
            self.main = lambda: calls.append(name)

    monkeypatch.setattr(importlib, "import_module", lambda name: Fake(name))
    timings = run_stages([("a", "mod.a"), ("b", "mod.b")])
    assert calls == ["mod.a", "mod.b"]
    assert list(timings.index) == ["a", "b"]


def test_compare_with_lock_uses_the_first_official_run(tmp_path):
    f = tmp_path / "model.json"
    f.write_text("tested", encoding="utf-8")

    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "runs": [
                    {"fingerprints": {"scorecard": sha256(f)}},
                    {"fingerprints": {"scorecard": "forced-rerun-hash"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert compare_with_lock(lock, {"scorecard": f})["match"].all()
    f.write_text("changed", encoding="utf-8")
    assert not compare_with_lock(lock, {"scorecard": f})["match"].any()
