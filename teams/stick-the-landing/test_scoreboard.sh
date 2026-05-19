#!/usr/bin/env bash
# Smoke test for the scoreboard pipeline.
#
# Does NOT run docker. Instead, it stubs evaluate.py output JSONs into the
# results dir layout that eval_all.sh would produce, then runs the aggregator
# directly via the same python heredoc by sourcing eval_all.sh with a
# short-circuit env var.
#
# Pass criteria:
#   * scoreboard.md exists and has these sections: "Aggregate per scenario",
#     "Per-cell breakdown"
#   * scoreboard.json is valid JSON with per_scenario keys for each scenario
#   * regression detection flags an injected -10% drop and exits non-zero
#   * improvement detection flags an injected +10% bump

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEAM_DIR="teams/stick-the-landing"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

RUN_DIR="$TMP/multi_test"
mkdir -p "$RUN_DIR"

# Helper: write a stub agent JSON
stub() {
    local scenario="$1" seed="$2" score="$3" outcome="${4:-LANDED}"
    cat > "$RUN_DIR/agent_${scenario}_seed${seed}.json" <<JSON
{
  "score": $score,
  "outcome": "$outcome",
  "scenario_id": "$scenario",
  "seed": $seed,
  "landing_position_error_m": 0.42,
  "max_descent_velocity_mps": 0.8,
  "time_to_terminate_s": 12.3,
  "latency_p95_ms": 8.5
}
JSON
}

# Baseline: easy across 3 seeds
stub easy 42 95
stub easy 1 90
stub easy 100 92

# Build a prev scoreboard for regression check
PREV_JSON="$TMP/prev_scoreboard.json"
cat > "$PREV_JSON" <<'JSON'
{
  "run_tag": "prev_test",
  "scenarios": ["easy"],
  "seeds": ["42", "1", "100"],
  "cells": {
    "easy/42": {"score": 100.0, "outcome": "LANDED"},
    "easy/1":  {"score": 85.0,  "outcome": "LANDED"},
    "easy/100":{"score": 90.0,  "outcome": "LANDED"}
  }
}
JSON
# easy/42: 100->95 = -5%  (boundary, not flagged at 5%)
# easy/1:  85->90  = +5.9% (improvement)
# easy/100:90->92  = +2.2% (neutral)

SB_MD="$TMP/scoreboard.md"
SB_JSON="$TMP/scoreboard.json"

# Run the aggregator only (extract from eval_all.sh would be brittle — call
# python directly with the same logic by sourcing the script in dry mode).
# Easier: run the aggregator inline mirroring eval_all.sh's heredoc.
python3 - "$RUN_DIR" "$SB_MD" "$SB_JSON" \
    "42 1 100" "easy" "test_tag" \
    "$PREV_JSON" "5.0" "$RUN_DIR/sim_score.json" <<'PY'
# Re-implementation NOT needed — instead, exec the aggregator from eval_all.sh
# by reading the heredoc out of the script and running it. For test simplicity,
# we re-run the aggregator block by sourcing a stripped copy.
import subprocess, sys, os, re
script = open("teams/stick-the-landing/eval_all.sh").read()
# Extract the python heredoc bounded by <<'PY' ... PY
m = re.search(r"python3 - .*?<<'PY'\n(.*?)\nPY\n", script, flags=re.S)
if not m:
    print("FAIL: could not find aggregator heredoc", file=sys.stderr)
    sys.exit(2)
agg = m.group(1)
# Execute the aggregator with our argv
ns = {"__name__": "__main__"}
old_argv = sys.argv
sys.argv = ["agg"] + sys.argv[1:]
try:
    exec(compile(agg, "<aggregator>", "exec"), ns)
except SystemExit as e:
    sys.exit(e.code)
finally:
    sys.argv = old_argv
PY
AGG_RC=$?

echo "aggregator exit: $AGG_RC"

# Validate outputs
FAIL=0
if [[ ! -f "$SB_MD" ]]; then
    echo "FAIL: scoreboard.md not produced"; FAIL=1
fi
if [[ ! -f "$SB_JSON" ]]; then
    echo "FAIL: scoreboard.json not produced"; FAIL=1
fi
for sec in "Aggregate per scenario" "Per-cell breakdown"; do
    if ! grep -q "$sec" "$SB_MD"; then
        echo "FAIL: scoreboard.md missing section '$sec'"; FAIL=1
    fi
done
python3 - "$SB_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
ok = True
for k in ["per_scenario", "cells", "scenarios", "seeds", "run_tag"]:
    if k not in d:
        print(f"FAIL: scoreboard.json missing key {k}")
        ok = False
if "easy" not in d.get("per_scenario", {}):
    print("FAIL: per_scenario.easy missing"); ok = False
ps = d["per_scenario"]["easy"]
for k in ["mean", "std", "min", "max", "land_rate"]:
    if k not in ps:
        print(f"FAIL: per_scenario.easy missing {k}"); ok = False
# Expect mean = (95+90+92)/3 = 92.33
if abs(ps["mean"] - (95+90+92)/3) > 0.01:
    print(f"FAIL: per_scenario.easy.mean != 92.33 ({ps['mean']})"); ok = False
sys.exit(0 if ok else 1)
PY
[[ $? -ne 0 ]] && FAIL=1

# Test improvement flagged (>5% bump on easy/1)
if ! grep -qE "Improvements" "$SB_MD"; then
    echo "FAIL: improvement section missing (expected easy/seed=1 +5.9%)"; FAIL=1
fi

# Now test regression detection: drop easy/42 to 80 (vs prev 100 = -20%)
stub easy 42 80
set +e
python3 - "$RUN_DIR" "$SB_MD" "$SB_JSON" \
    "42 1 100" "easy" "test_tag2" \
    "$PREV_JSON" "5.0" "$RUN_DIR/sim_score.json" <<'PY'
import sys, re
script = open("teams/stick-the-landing/eval_all.sh").read()
m = re.search(r"python3 - .*?<<'PY'\n(.*?)\nPY\n", script, flags=re.S)
agg = m.group(1)
sys.argv = ["agg"] + sys.argv[1:]
try:
    exec(compile(agg, "<aggregator>", "exec"), {"__name__": "__main__"})
except SystemExit as e:
    sys.exit(e.code)
PY
REGRESSION_RC=$?
set -e
if [[ "$REGRESSION_RC" -ne 1 ]]; then
    echo "FAIL: expected aggregator to exit 1 on regression, got $REGRESSION_RC"; FAIL=1
fi
if ! grep -qE "Regression" "$SB_MD"; then
    echo "FAIL: regression section missing"; FAIL=1
fi

if [[ $FAIL -ne 0 ]]; then
    echo
    echo "=== scoreboard.md (failed test) ==="
    cat "$SB_MD" || true
    echo "=== scoreboard.json (failed test) ==="
    cat "$SB_JSON" || true
    exit 1
fi

echo "OK: scoreboard smoke test passed"
