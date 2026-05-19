#!/usr/bin/env bash
# Single-scenario quick check (~30s). For teammates to run locally before
# pushing. Default: EASY @ seed 42. Output goes to stdout (JSON) + summary.
#
# Usage:
#   teams/stick-the-landing/run_eval.sh                 # easy seed=42
#   teams/stick-the-landing/run_eval.sh medium 7        # medium seed=7
#   teams/stick-the-landing/run_eval.sh hard            # hard seed=42
#   CHALLENGE_IMAGE=challenge-eval:full teams/stick-the-landing/run_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEAM_DIR="teams/stick-the-landing"
SCENARIO="${1:-easy}"
SEED="${2:-42}"
AGENT="$TEAM_DIR/agent.py"
DRONE_SIM="$TEAM_DIR/drone_sim.py"
DRONE="${DRONE:-vtol}"

USE_REF_FLAG=""
if [[ "${CHALLENGE_IMAGE:-}" == *":full"* ]]; then
    USE_REF_FLAG="--use-reference-sim"
fi

OUT="$(mktemp /tmp/run_eval_XXXXXX.json)"
trap 'rm -f "$OUT"' EXIT

echo "==> $SCENARIO seed=$SEED image=${CHALLENGE_IMAGE:-challenge-eval:base}" >&2
./docker/run-local.sh python evaluation/evaluate.py \
    --agent     "$AGENT" \
    --drone-sim "$DRONE_SIM" \
    --drone     "$DRONE" \
    --scenario  "$SCENARIO" \
    --headless --quiet --seed "$SEED" \
    $USE_REF_FLAG \
    --output    "$OUT" \
    > /dev/null

python3 - "$OUT" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(json.dumps(d, indent=2))
sys.stderr.write(
    f"\n  outcome:  {d.get('outcome')}\n"
    f"  score:    {d.get('score')}/115\n"
    f"  land_err: {d.get('landing_position_error_m')}\n"
    f"  desc_v:   {d.get('max_descent_velocity_mps')}\n"
    f"  t_land:   {d.get('time_to_terminate_s')}\n"
    f"  p95_ms:   {d.get('latency_p95_ms')}\n"
)
PY
