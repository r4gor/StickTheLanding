#!/usr/bin/env bash
# Multi-seed scoreboard for the stick-the-landing submission.
#
# Runs the team's agent.py + drone_sim.py through one or more scenarios
# across one or more seeds, then aggregates into:
#   * teams/stick-the-landing/results/<TAG>/agent_<scenario>_seed<seed>.json
#   * teams/stick-the-landing/scoreboard.md  (timestamped, human + machine readable)
#   * teams/stick-the-landing/scoreboard.json (machine-readable mirror)
#
# Regression sentinel: if PREV_SCOREBOARD points to a previous scoreboard.json,
# and any (scenario, seed) score drops by more than REGRESSION_PCT (default 5%),
# the script exits 1 with a red marker in the markdown.
#
# Usage:
#   teams/stick-the-landing/eval_all.sh
#       # default: SEEDS="42 1 100" SCENARIOS="easy medium hard"
#   SEEDS="42" SCENARIOS="easy" teams/stick-the-landing/eval_all.sh
#       # fast single-seed smoke
#   PARALLEL=4 teams/stick-the-landing/eval_all.sh
#       # run up to 4 (scenario, seed) jobs concurrently via xargs -P
#   SKIP_SIM=1 teams/stick-the-landing/eval_all.sh
#       # skip simulator-quality scorer (faster)
#   PREV_SCOREBOARD=teams/stick-the-landing/scoreboard.json \
#       teams/stick-the-landing/eval_all.sh
#       # compare against prior scoreboard and flag regressions
#   CHALLENGE_IMAGE=challenge-eval:full teams/stick-the-landing/eval_all.sh
#       # score against organizer reference sim
#
# Runs everything inside Docker via `./docker/run-local.sh` (no local Python deps).

set -euo pipefail

# cd to repo root regardless of where the script is invoked from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEAM_DIR="teams/stick-the-landing"
AGENT="$TEAM_DIR/agent.py"
DRONE_SIM="$TEAM_DIR/drone_sim.py"
SUBMISSION="$TEAM_DIR/submission.yaml"
DRONE="${DRONE:-vtol}"
SEEDS="${SEEDS:-42 1 100}"
SCENARIOS="${SCENARIOS:-easy medium hard}"
PARALLEL="${PARALLEL:-1}"
SKIP_SIM="${SKIP_SIM:-0}"
REGRESSION_PCT="${REGRESSION_PCT:-5.0}"
PREV_SCOREBOARD="${PREV_SCOREBOARD:-}"
RESULTS_DIR="$TEAM_DIR/results"
RUN_TAG="multi_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_TAG"
SCOREBOARD_MD="$TEAM_DIR/scoreboard.md"
SCOREBOARD_JSON="$TEAM_DIR/scoreboard.json"

mkdir -p "$RUN_DIR"

# If the organizer :full image is requested, evaluate.py needs --use-reference-sim
USE_REF_FLAG=""
if [[ "${CHALLENGE_IMAGE:-}" == *":full"* ]]; then
    USE_REF_FLAG="--use-reference-sim"
fi

echo "=========================================================="
echo "  stick-the-landing scoreboard"
echo "    seeds:     $SEEDS"
echo "    scenarios: $SCENARIOS"
echo "    parallel:  $PARALLEL"
echo "    image:     ${CHALLENGE_IMAGE:-challenge-eval:base}"
echo "    results:   $RUN_DIR"
[[ -n "$PREV_SCOREBOARD" ]] && echo "    prev:      $PREV_SCOREBOARD  (regression>$REGRESSION_PCT%)"
echo "=========================================================="

# ---- 1. Build the (scenario, seed) job list ----------------------------
JOBS_FILE="$RUN_DIR/.jobs"
: > "$JOBS_FILE"
for s in $SCENARIOS; do
    for sd in $SEEDS; do
        echo "$s $sd" >> "$JOBS_FILE"
    done
done
JOB_COUNT=$(wc -l < "$JOBS_FILE" | tr -d ' ')
echo "  jobs: $JOB_COUNT"

# ---- 2. Define the runner --------------------------------------------
run_one() {
    local scenario="$1"
    local seed="$2"
    local out="$RUN_DIR/agent_${scenario}_seed${seed}.json"
    local log="$RUN_DIR/agent_${scenario}_seed${seed}.log"
    echo "  [$(date +%H:%M:%S)] -> scenario=$scenario seed=$seed"
    if ./docker/run-local.sh python evaluation/evaluate.py \
            --agent      "$AGENT" \
            --drone-sim  "$DRONE_SIM" \
            --drone      "$DRONE" \
            --scenario   "$scenario" \
            --headless --quiet --seed "$seed" \
            $USE_REF_FLAG \
            --output     "$out" \
            > "$log" 2>&1; then
        echo "  [$(date +%H:%M:%S)] <- $scenario seed=$seed OK"
    else
        echo "  [$(date +%H:%M:%S)] <- $scenario seed=$seed FAILED (see $log)" >&2
    fi
}
export -f run_one
export RUN_DIR AGENT DRONE_SIM DRONE USE_REF_FLAG

# ---- 3. Execute (parallel or serial) ---------------------------------
if [[ "$PARALLEL" -gt 1 ]]; then
    # xargs -P spawns one bash per line, calling run_one with two args
    < "$JOBS_FILE" xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ || true
else
    while read -r scenario seed; do
        run_one "$scenario" "$seed"
    done < "$JOBS_FILE"
fi

# ---- 4. Optional: simulator-quality score (single run, no seeds) -----
SIM_JSON="$RUN_DIR/sim_score.json"
if [[ "$SKIP_SIM" != "1" ]]; then
    echo
    echo "--- simulator-quality ---"
    ./docker/run-local.sh python evaluation/sim_scorer.py \
        --drone-sim   "$DRONE_SIM" \
        --drone       "$DRONE" \
        --submission  "$SUBMISSION" \
        --output      "$SIM_JSON" \
        > "$RUN_DIR/sim_score.log" 2>&1 \
        || echo "  sim_scorer FAILED (see $RUN_DIR/sim_score.log)" >&2
fi

# ---- 5. Aggregate into scoreboard.{md,json} --------------------------
echo
echo "=========================================================="
echo "  AGGREGATING SCOREBOARD"
echo "=========================================================="

python3 - "$RUN_DIR" "$SCOREBOARD_MD" "$SCOREBOARD_JSON" \
    "$SEEDS" "$SCENARIOS" "$RUN_TAG" \
    "$PREV_SCOREBOARD" "$REGRESSION_PCT" "$SIM_JSON" <<'PY'
import json, os, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
md_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
seeds = sys.argv[4].split()
scenarios = sys.argv[5].split()
run_tag = sys.argv[6]
prev_path = sys.argv[7]
regression_pct = float(sys.argv[8])
sim_path = Path(sys.argv[9])

# Load all per-run JSONs
runs = {}  # (scenario, seed) -> result dict
for s in scenarios:
    for sd in seeds:
        p = run_dir / f"agent_{s}_seed{sd}.json"
        if p.exists():
            try:
                with p.open() as f:
                    runs[(s, sd)] = json.load(f)
            except Exception as e:
                runs[(s, sd)] = {"error": str(e), "outcome": "ERROR", "score": 0.0}
        else:
            runs[(s, sd)] = {"outcome": "MISSING", "score": 0.0}

# Load previous scoreboard for regression check
prev = None
if prev_path and Path(prev_path).is_file():
    try:
        with open(prev_path) as f:
            prev = json.load(f)
    except Exception as e:
        print(f"warn: could not load prev scoreboard {prev_path}: {e}", file=sys.stderr)

def pct_delta(curr, p):
    if p is None or p == 0:
        return None
    return 100.0 * (curr - p) / abs(p)

# Aggregate per scenario
per_scenario = {}
for s in scenarios:
    scores = []
    landings = 0
    for sd in seeds:
        r = runs.get((s, sd), {})
        sc = r.get("score")
        if sc is not None:
            scores.append(float(sc))
        if r.get("outcome") == "LANDED":
            landings += 1
    if scores:
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        smin = min(scores)
        smax = max(scores)
    else:
        mean = std = smin = smax = 0.0
    per_scenario[s] = {
        "mean": mean, "std": std, "min": smin, "max": smax,
        "land_rate": landings / max(len(seeds), 1),
        "scores_by_seed": {sd: runs.get((s, sd), {}).get("score") for sd in seeds},
        "outcomes_by_seed": {sd: runs.get((s, sd), {}).get("outcome", "MISSING") for sd in seeds},
    }

# Regression detection per (scenario, seed)
regressions = []
improvements = []
if prev and "cells" in prev:
    for s in scenarios:
        for sd in seeds:
            key = f"{s}/{sd}"
            curr = runs.get((s, sd), {}).get("score")
            p = prev["cells"].get(key, {}).get("score")
            if curr is None or p is None:
                continue
            d = pct_delta(curr, p)
            if d is None:
                continue
            if d < -regression_pct:
                regressions.append((s, sd, p, curr, d))
            elif d > regression_pct:
                improvements.append((s, sd, p, curr, d))

# Sim score
sim_data = {}
if sim_path.is_file():
    try:
        with sim_path.open() as f:
            sim_data = json.load(f)
    except Exception:
        pass

# Build scoreboard.json
out_json = {
    "run_tag": run_tag,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "scenarios": scenarios,
    "seeds": seeds,
    "per_scenario": per_scenario,
    "cells": {
        f"{s}/{sd}": {
            "score": runs.get((s, sd), {}).get("score"),
            "outcome": runs.get((s, sd), {}).get("outcome", "MISSING"),
            "landing_position_error_m": runs.get((s, sd), {}).get("landing_position_error_m"),
            "max_descent_velocity_mps": runs.get((s, sd), {}).get("max_descent_velocity_mps"),
            "time_to_terminate_s": runs.get((s, sd), {}).get("time_to_terminate_s"),
            "latency_p95_ms": runs.get((s, sd), {}).get("latency_p95_ms"),
        }
        for s in scenarios for sd in seeds
    },
    "sim_quality": {
        "total_score": sim_data.get("total_score"),
        "tier_0_passed": sim_data.get("tier_0_passed"),
        "tier_1_score": sim_data.get("tier_1_score"),
    } if sim_data else None,
    "regressions": [
        {"scenario": s, "seed": sd, "prev": p, "curr": c, "pct": d}
        for s, sd, p, c, d in regressions
    ],
    "improvements": [
        {"scenario": s, "seed": sd, "prev": p, "curr": c, "pct": d}
        for s, sd, p, c, d in improvements
    ],
    "prev_scoreboard": prev_path or None,
    "regression_pct_threshold": regression_pct,
}

json_path.write_text(json.dumps(out_json, indent=2, default=str))

# Build scoreboard.md
def fmt(x, prec=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

md = []
md.append(f"# stick-the-landing scoreboard")
md.append("")
md.append(f"- **run tag**: `{run_tag}`")
md.append(f"- **timestamp**: {out_json['timestamp_utc']}")
md.append(f"- **scenarios**: {' · '.join(scenarios)}")
md.append(f"- **seeds**: {' · '.join(seeds)}")
md.append(f"- **image**: `{os.environ.get('CHALLENGE_IMAGE', 'challenge-eval:base')}`")
if prev_path:
    md.append(f"- **vs prev**: `{prev_path}` (threshold {regression_pct}%)")
md.append("")

# Regression / improvement banner
if regressions:
    md.append(f"## ⛔ Regressions ({len(regressions)})")
    md.append("")
    md.append("| scenario | seed | prev | curr | Δ% |")
    md.append("|---|---|---|---|---|")
    for s, sd, p, c, d in regressions:
        md.append(f"| {s} | {sd} | {fmt(p)} | {fmt(c)} | 🔴 {d:+.1f}% |")
    md.append("")
if improvements:
    md.append(f"## ✅ Improvements ({len(improvements)})")
    md.append("")
    md.append("| scenario | seed | prev | curr | Δ% |")
    md.append("|---|---|---|---|---|")
    for s, sd, p, c, d in improvements:
        md.append(f"| {s} | {sd} | {fmt(p)} | {fmt(c)} | 🟢 {d:+.1f}% |")
    md.append("")

# Per-scenario aggregate
md.append("## Aggregate per scenario")
md.append("")
md.append("| scenario | mean | std | min | max | land rate |")
md.append("|---|---|---|---|---|---|")
for s in scenarios:
    ps = per_scenario[s]
    md.append(f"| **{s}** | {fmt(ps['mean'])} | {fmt(ps['std'])} | {fmt(ps['min'])} | {fmt(ps['max'])} | {int(ps['land_rate']*len(seeds))}/{len(seeds)} |")
md.append("")

# Per-cell breakdown
md.append("## Per-cell breakdown")
md.append("")
header = ["scenario", "seed", "outcome", "score", "land_err_m", "desc_vps", "t_land_s", "p95_ms"]
md.append("| " + " | ".join(header) + " |")
md.append("|" + "|".join(["---"] * len(header)) + "|")
for s in scenarios:
    for sd in seeds:
        r = runs.get((s, sd), {})
        cell_delta = ""
        if prev and "cells" in prev:
            p = prev["cells"].get(f"{s}/{sd}", {}).get("score")
            sc = r.get("score")
            if p is not None and sc is not None and p != 0:
                d = 100.0 * (sc - p) / abs(p)
                if d < -regression_pct:
                    cell_delta = f" 🔴{d:+.1f}%"
                elif d > regression_pct:
                    cell_delta = f" 🟢{d:+.1f}%"
                else:
                    cell_delta = f" ({d:+.1f}%)"
        score_cell = f"{fmt(r.get('score'))}{cell_delta}"
        md.append("| " + " | ".join([
            s, sd, r.get("outcome", "MISSING"),
            score_cell,
            fmt(r.get("landing_position_error_m")),
            fmt(r.get("max_descent_velocity_mps")),
            fmt(r.get("time_to_terminate_s")),
            fmt(r.get("latency_p95_ms"), 1),
        ]) + " |")
md.append("")

# Sim quality
if sim_data:
    md.append("## Simulator quality (tier-0 / tier-1)")
    md.append("")
    md.append(f"- total: **{sim_data.get('total_score', 0)}/30**")
    md.append(f"- tier_0_passed: {sim_data.get('tier_0_passed')}")
    md.append(f"- tier_1_score: {sim_data.get('tier_1_score', 0)}")
    failed_t1 = [t.get("feature", "?") for t in sim_data.get("tier_1", []) if not t.get("passed")]
    if failed_t1:
        md.append(f"- tier_1 not earned: {', '.join(failed_t1)}")
    md.append("")

md.append("---")
md.append(f"_artifacts: `{run_dir}` · json mirror: `{json_path}`_")
md.append("")

md_path.write_text("\n".join(md))

# Console summary
print()
print(f"scoreboard.md  -> {md_path}")
print(f"scoreboard.json -> {json_path}")
print()
print("Aggregate:")
for s in scenarios:
    ps = per_scenario[s]
    print(f"  {s:<8}  mean={ps['mean']:.2f}  std={ps['std']:.2f}  "
          f"min={ps['min']:.2f}  max={ps['max']:.2f}  "
          f"landed={int(ps['land_rate']*len(seeds))}/{len(seeds)}")
if sim_data:
    print(f"  sim     total={sim_data.get('total_score')}/30")
if regressions:
    print(f"\n⛔ {len(regressions)} regression(s) > {regression_pct}%:")
    for s, sd, p, c, d in regressions:
        print(f"   {s}/seed{sd}: {p:.2f} -> {c:.2f}  ({d:+.1f}%)")
if improvements:
    print(f"\n✅ {len(improvements)} improvement(s) > {regression_pct}%")

# Exit code = 1 iff regressions detected
sys.exit(1 if regressions else 0)
PY

RC=$?
echo
echo "JSON breakdowns saved to: $RUN_DIR"
echo "scoreboard.md            : $SCOREBOARD_MD"
echo "scoreboard.json          : $SCOREBOARD_JSON"
exit $RC
