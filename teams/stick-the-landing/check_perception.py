"""Ground-truth dev harness for the perception stage — DEV ONLY.

Flies the baseline agent through each scenario, calls the team agent's
perceive() + marker_to_world() every step, and compares the recovered marker
position to the env's ground-truth boat position.

This script reads info["boat_position"] / info["boat_heading"] — LEGAL in a
dev/test script, FORBIDDEN in the submitted agent (which must infer the boat
pose from the camera alone).

Run inside Docker, from the repo root:
    ./docker/run-local.sh python teams/stick-the-landing/check_perception.py
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from boat_landing.env import BoatLandingEnv         # noqa: E402
from agents.agent_baseline import BaselineAgent     # noqa: E402
from agent import Agent, marker_to_world            # noqa: E402


SCENARIOS = ["easy", "medium", "hard"]
SEED = 1


def run_scenario(name: str, seed: int = SEED) -> dict:
    """Fly one scenario with the baseline agent, scoring perception each step."""
    scenario_path = os.path.join(REPO_ROOT, "scenarios", f"{name}.yaml")
    env = BoatLandingEnv(scenario_path)
    try:
        flyer = BaselineAgent(env.drone_spec)   # known-good pilot
        perc = Agent(env.drone_spec)            # the agent under test
        obs, info = env.reset(seed=seed)

        errs, yaw_errs = [], []
        n_steps = n_det = n_fresh = 0
        done = False
        while not done:
            p = perc.perceive(obs["camera"])
            n_steps += 1
            if p["detected"]:
                n_det += 1
                if not p["stale"]:
                    n_fresh += 1
                    mw, myaw = marker_to_world(p, obs["state"])
                    gt = np.asarray(info["boat_position"], dtype=np.float64)
                    errs.append(float(np.linalg.norm(mw[:2] - gt[:2])))
                    # heading gap, collapsed to [0, pi/2] (landing is mod pi)
                    dh = (myaw - float(info["boat_heading"])) % np.pi
                    yaw_errs.append(float(min(dh, np.pi - dh)))
            obs, _, terminated, truncated, info = env.step(flyer.act(obs))
            done = terminated or truncated

        return {
            "scenario": name,
            "steps": n_steps,
            "detect_rate": n_det / max(n_steps, 1),
            "fresh": n_fresh,
            "xy_rms_m": float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"),
            "xy_p95_m": float(np.percentile(errs, 95)) if errs else float("nan"),
            "yaw_mean_deg": float(np.degrees(np.mean(yaw_errs))) if yaw_errs else float("nan"),
        }
    finally:
        env.close()


def main() -> int:
    print(f"{'scenario':9s} {'steps':>6s} {'detect':>7s} {'fresh':>6s} "
          f"{'xy_rms':>9s} {'xy_p95':>9s} {'yaw_mean':>9s}")
    print("-" * 62)
    results = []
    for name in SCENARIOS:
        try:
            r = run_scenario(name)
        except Exception:  # noqa: BLE001 — one bad scenario must not kill the run
            print(f"{name:9s}  ERROR", file=sys.stderr)
            traceback.print_exc()
            continue
        results.append(r)
        print(f"{r['scenario']:9s} {r['steps']:6d} {r['detect_rate']:6.2f} "
              f"{r['fresh']:6d} {r['xy_rms_m']:8.3f}m {r['xy_p95_m']:8.3f}m "
              f"{r['yaw_mean_deg']:8.2f}d")

    print()
    print("yaw_mean is the marker-vs-boat heading gap; its near-constant part")
    print("is the calibration offset to hand to Person 3 (Control).")

    easy = next((r for r in results if r["scenario"] == "easy"), None)
    if easy is None or not (easy["detect_rate"] > 0.3):
        print("\nperception harness: FAIL — EASY detection rate too low / no run.")
        return 1
    print("\nperception harness: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
