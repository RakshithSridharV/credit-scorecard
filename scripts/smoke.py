"""Smoke test a running scoring API (D-026). Standard library only, so it runs
anywhere: on the host against Docker, or later in CI.

    python scripts/smoke.py [base_url]      # default http://127.0.0.1:8000

Exits non-zero on the first failed check.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 4000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}
RISKY = {
    **APPLICANT,
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "NumberOfTimes90DaysLate": 1,
    "age": 28,
    "DebtRatio": 0.8,
    "MonthlyIncome": 3000,
}


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name: str, ok: bool, detail: object) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        sys.exit(1)


def wait_until_ready(timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if call("GET", "/ready")[0] == 200:
                return
        except OSError:
            pass  # not listening yet
        time.sleep(1)
    check("ready within timeout", False, f"{timeout_s}s")


if __name__ == "__main__":
    wait_until_ready()
    status, health = call("GET", "/health")
    check("/health", status == 200 and health["cutoff"] == 549, health)
    status, ok = call("POST", "/score", APPLICANT)
    check("approve", status == 200 and ok["decision"] == "approve", (ok["score"], ok["decision"]))
    status, bad = call("POST", "/score", RISKY)
    codes = [r["code"] for r in bad.get("reasons", [])]
    check("decline + reasons", status == 200 and bad["decision"] == "decline" and codes, codes)
    status, err = call("POST", "/score", {**APPLICANT, "age": 17})
    check("age 17 -> 422", status == 422, err["detail"][0]["msg"])
    check("ids differ", ok["decision_id"] != bad["decision_id"], bad["decision_id"])
    print("smoke test passed")
