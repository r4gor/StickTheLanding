# Stick the Landing — team proposal (read first)

This is the **1-page version**. Drill into the per-stage playbooks (`perceive.md`, `estimate.md`, `decide.md`, `control.md`) and the contract spec (`contracts.md`) for the details.

---

## Score map (verified against `scorer.py`)

```
LANDED → score = 50 · precision · time_factor · battery_factor
              + 5  (soft landing, STRICT max_descent_velocity < 1.0 m/s)
              + up to 15 (estimation, RMSE of 2D XY position; linear 0m→15pts, 2m→0pts)
              + 15 (FC-compatible:   agent.act_setpoint method exists)
              + 15 (latency p95 ≤ 20 ms — already 1ms baseline, just don't regress)
              + 15 (recovery:        get_last_estimate keeps working during 1s camera blackout)
CRASHED   → −20
TIMEOUT/etc → 0
```

**Baseline today (verified by running it):** ~69/70 on EASY, **times out on MEDIUM and HARD**. So almost all the headroom is on the harder scenarios + the four HW-readiness paths.

---

## Team split

| Person | Stage | Output |
|---|---|---|
| **Ludo (aerospace)** | `drone_sim.py` Tier-1 features + boat state-space spec | Sim track (+~15 pts) + KF dynamics for AI/ML |
| **AI/ML dev** | `estimate()` — Kalman filter | Biggest agent-track unlock. Unlocks recovery + estimation bonus |
| **CS dev #1** | `decide()` — state machine | Hysteresis + abort + soft-landing gate |
| **CS dev #2** | `control()` — refactor + FF + yaw + soft descent + **`act_setpoint`** | Biggest single-line score win (+15 fc_compat). Also the eval harness owner. |
| **CS dev #2 (1 hr)** | Eval harness (sweep scenarios × seeds, dashboard) | Without this, nobody knows if a PR helped |
| **Ludo (last 60-90 min)** | HW track — concept pitch + CAD sketch | Human-judged (60 pts possible) |

---

## How merges stay additive

Every stage takes a dict and returns a dict. Every reader uses `.get(key, default)`. New optional fields are always OK. **See `contracts.md` for the locked schemas.** Three structural decisions worth knowing:

1. **`history=None`** is being dropped from `estimate()` — KF state lives on `self`.
2. **`decide()` returns a dict**, not a string, so it can pass tuning to `control()` (target descent rate, yaw lock flag, commit_soft).
3. **`control()` is split internally** into `_control_setpoints()` (returns 4-tuple) + the mixer call. `act_setpoint()` re-uses the setpoints path and is the entire +15 FC-compat bonus.

---

## Critical-path commits (in suggested order)

1. **`act_setpoint()` refactor** (30 min, CS#2) — +15 pts, smallest diff. Ship first to establish HW-readiness baseline.
2. **KF velocity output** (2-3 h, AI/ML) — populate `boat_estimate.velocity` non-zero. Single biggest score delta.
3. **Hysteresis + frame debounce in `decide()`** (1 h, CS#1) — 15-30% median-score lift, kills flicker failures.
4. **Velocity feedforward in `control()`** (1-2 h, CS#2) — 30-50% lateral RMS reduction. Needs (2).
5. **Two-stage approach gating in `decide()`** (30 min, CS#1) — MBZIRC standard pattern. Needs (3).
6. **Soft-landing profile in `control()`** (1 h, CS#2) — +5 pts when `commit_soft` is True. Needs (3).
7. **Active yaw + lock-and-freeze** (2 h, CS#2 + CS#1) — unlocks fuselage-aligned landing condition. Needs KF heading from (2).
8. **Sim Tier-1: ground_effect + aero_drag + battery_sag** (3-4 h, Ludo) — +15 pts on sim track.
9. **Recovery bonus** (already free via (2)'s predict-only KF) — +15 if private scenarios enable it.

---

## What to skip (high confidence, after web research)

- **MPC** — TinyMPC/Acados RTI need significant integration time; FF + exponential descent get ~80% of the gain.
- **SE(3) geometric control / cascade rewrite** — current cascade matches PX4 and ArduPilot.
- **Particle filter / SE(2) EKF** — measurement is unimodal Gaussian after PnP; linear KF wins.
- **Body-rate output instead of attitude-setpoint** — scorer just checks method presence, not output format.
- **Refining the baseline `act()` method** — frozen in `agent_template.py:87-101`. Override the 4 stages instead.
- **Modifying `drone_sim.py` cosmetically** — anti-bluff AST check vs `agents/drone_sim_baseline.py` (comments/whitespace stripped). Renames don't help; need real physics additions.

---

## What's already in the codebase that the team might miss

- **`DefaultAttitudeController` already does gyroscopic precession feedforward** when `state.motor_omegas` is populated. Don't reinvent.
- **Yaw alignment is mod π** (both 0° and 180° pass). Tolerances: 35°/30°/25° for EASY/MEDIUM/HARD.
- **Yaw alignment is checked in boat body frame**, not world — so a corner landing on a rotated boat still passes.
- **Battery has an "aggressive drain" mode** when motors are working hard (not active in public scenarios but may be in private).
- **HARD is gentler than the README suggested**: 1.8 m/s curved (radius 30 m), 15 fps camera, fog density 0.10. Centripetal accel is only ~0.11 m/s² — KF Q values don't need to be huge.
- **The scorer **forces NaN inputs to worst-case** — emit `None` instead of NaN/Inf at every layer.
- **The reference simulator is x86_64 Linux only** — `--platform linux/amd64` runs it on Apple Silicon under Rosetta (3.43× slower per `benchmark.py`).

---

## Read order for new teammates

1. **`proposal.md`** (this file) — score map + team split.
2. **`contracts.md`** — what every stage in/out looks like; merge rules.
3. **Your stage playbook** (one of `perceive.md`, `estimate.md`, `decide.md`, `control.md`) — ranked work for your stage.
4. **`agent_pipeline.md`** — math for the 4-stage agent pipeline (reference).
5. **`sim_pipeline.md`** — sim contract + math for the 9-step `drone_sim.step()` (reference, only if you're touching sim).
6. Skim the public docs in `docs/` to confirm anything that's not in here.
