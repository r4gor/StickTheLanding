#!/usr/bin/env bash
# Comprehensive eval for the stick-the-landing submission. Runs the team's
# agent.py + drone_sim.py through every public scenario (easy/medium/hard)
# via Docker, then runs the simulator-quality scorer once, and prints a
# combined breakdown.
#
# Usage:
#   teams/stick-the-landing/eval_all.sh                  # default seed, all scenarios
#   SEED=7  teams/stick-the-landing/eval_all.sh          # different seed
#   SCENARIOS="easy hard" teams/stick-the-landing/eval_all.sh
#   CHALLENGE_IMAGE=challenge-eval:full teams/stick-the-landing/eval_all.sh
#                                                       # score against organizer reference sim
#
# Runs everything inside Docker via `./docker/run-local.sh` — no local Python
# install required. JSON outputs are saved under teams/stick-the-landing/results/.

set -euo pipefail

# cd to repo root regardless of where the script is invoked from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEAM_DIR="teams/stick-the-landing"
AGENT="$TEAM_DIR/agent.py"
DRONE_SIM="$TEAM_DIR/drone_sim.py"
SUBMISSION="$TEAM_DIR/submission.yaml"
DRONE="${DRONE:-vtol}"
SEED="${SEED:-42}"
SCENARIOS="${SCENARIOS:-easy medium hard}"
RESULTS_DIR="$TEAM_DIR/results"
RUN_TAG="seed${SEED}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_TAG"

mkdir -p "$RUN_DIR"

# If the organizer :full image is requested, evaluate.py needs --use-reference-sim
# (otherwise it runs against your local drone_sim.py and never sees the reference).
USE_REF_FLAG=""
if [[ "${CHALLENGE_IMAGE:-}" == *":full"* ]]; then
    USE_REF_FLAG="--use-reference-sim"
fi

echo "=========================================================="
echo "  stick-the-landing eval — seed=$SEED scenarios=($SCENARIOS)"
echo "  image: ${CHALLENGE_IMAGE:-challenge-eval:base}"
echo "  results: $RUN_DIR"
echo "=========================================================="

# ---- 1. Agent score, one run per scenario --------------------------------
for s in $SCENARIOS; do
    out="$RUN_DIR/agent_${s}.json"
    echo
    echo "--- scenario: $s ---"
    ./docker/run-local.sh python evaluation/evaluate.py \
        --agent      "$AGENT" \
        --drone-sim  "$DRONE_SIM" \
        --drone      "$DRONE" \
        --scenario   "$s" \
        --headless --quiet --seed "$SEED" \
        $USE_REF_FLAG \
        --output     "$out" \
        > /dev/null   # evaluate.py also dumps JSON to stdout; --output is the file we read
done

# ---- 2. Simulator-quality score, one run --------------------------------
sim_out="$RUN_DIR/sim_score.json"
echo
echo "--- simulator-quality ---"
./docker/run-local.sh python evaluation/sim_scorer.py \
    --drone-sim   "$DRONE_SIM" \
    --drone       "$DRONE" \
    --submission  "$SUBMISSION" \
    --output      "$sim_out" \
    > /dev/null

# ---- 3. Aggregate + pretty-print ----------------------------------------
echo
echo "=========================================================="
echo "  SUMMARY"
echo "=========================================================="
python3 - "$RUN_DIR" "$SCENARIOS" <<'PY'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
scenarios = sys.argv[2].split()

agent_total = 0.0
agent_landings = 0
print()
print(f"{'scenario':<10} {'outcome':<18} {'score/115':>10} {'land_err_m':>11} {'descent_mps':>12} {'t_land_s':>9} {'p95_ms':>8}")
print("-" * 84)
for s in scenarios:
    p = run_dir / f"agent_{s}.json"
    if not p.exists():
        print(f"{s:<10} (no result)")
        continue
    with p.open() as f:
        d = json.load(f)
    outcome = d.get("outcome", "?")
    score = d.get("score", 0.0)
    err = d.get("landing_position_error_m")
    dv = d.get("max_descent_velocity_mps")
    tt = d.get("time_to_terminate_s")
    p95 = d.get("latency_p95_ms")
    agent_total += float(score or 0.0)
    if outcome == "LANDED":
        agent_landings += 1
    fmt = lambda x, w, prec=2: ("-".rjust(w) if x is None else f"{x:>{w}.{prec}f}")
    print(f"{s:<10} {outcome:<18} {score:>10.2f} {fmt(err,11)} {fmt(dv,12)} {fmt(tt,9)} {fmt(p95,8,1)}")

print()
sim_p = run_dir / "sim_score.json"
if sim_p.exists():
    with sim_p.open() as f:
        sim = json.load(f)
    sim_total = sim.get("total_score", 0)
    print(f"  Simulator quality: {sim_total}/30   (tier_0_passed={sim.get('tier_0_passed')}, tier_1={sim.get('tier_1_score', 0)})")
    failed_t1 = [t.get("feature", "?") for t in sim.get("tier_1", []) if not t.get("passed")]
    if failed_t1:
        print(f"    tier_1 not earned: {', '.join(failed_t1)}")
else:
    sim_total = 0
    print("  Simulator quality: (no result)")

print()
print(f"  Agent score sum (3 scenarios, max 345 = 3×115):  {agent_total:.2f}")
print(f"  Agent mean / scenario:                            {agent_total/max(len(scenarios),1):.2f}/115")
print(f"  Scenarios landed:                                 {agent_landings}/{len(scenarios)}")
print(f"  Sim quality:                                      {sim_total}/30")
print()
print("  Note: HW-track (0-60) is human-judged at the event and not")
print("  included here. Total scoreboard formula in docs/CHALLENGE.md.")
PY

echo
echo "JSON breakdowns saved to: $RUN_DIR"
