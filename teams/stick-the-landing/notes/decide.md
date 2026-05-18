# Decision-layer playbook

**Owner suggestion:** one of the CS devs (probably the one who also owns the eval harness — there's overlap on understanding the scoring formula).

**Goal:** eliminate phase flicker, add abort/retry, prepare for the soft-landing bonus. Single biggest leverage: **hysteresis + frame debouncing typically moves median score 15-30%** by killing flicker-induced failures.

---

## Scoring math reality check (verified against `scorer.py`)

```
score = -20 if CRASHED
      = 0   if TIMEOUT / OUT_OF_BATTERY / WALL_TIMEOUT / ERROR / OOM / ABORTED
      = landing_score + soft_landing_bonus + estimation_bonus + (HW bonuses if LANDED)

landing_score   = 50 * precision * time_factor * battery_factor
precision       = max(0, 1 - landing_pos_error / 1.5)   # 0 when error ≥ 1.5 m
time_factor     = max(0.5, 1 - time_to_land / duration_max)   # floor 0.5
battery_factor  = max(0.3, battery_remaining)             # floor 0.3

soft_landing_bonus  = 5  if max_descent_velocity STRICTLY < 1.0 m/s   else 0
estimation_bonus    = 15 * (1 - min(rmse_xy, 2.0) / 2.0)
fc_compat_bonus     = 15 if agent.act_setpoint exists (LANDED only)
latency_bonus       = 15 if p95 ≤ 20 ms (LANDED only)
recovery_bonus      = 15 if recovery test passed (LANDED only, scenarios that enable it)
```

Two non-obvious points the decision layer should bake in:
1. **Soft landing is `< 1.0 m/s` (strict)**, not `≤`. So aim for ~0.3-0.5 m/s touchdown with margin.
2. **Time-factor floor is 0.5** — even a slow-but-correct landing keeps half the time multiplier.

---

## Scenario reality check (from `scenarios/*.yaml`)

| | EASY | MEDIUM | HARD |
|---|---|---|---|
| Boat motion | static | linear 1.0 m/s | curve 1.8 m/s, r=30 m |
| Wind mean force | 0 | [0.3, 0, 0] N | [0.6, -0.2, 0] N |
| Oscillation freq | 0 | 0.3 Hz | 0.5 Hz |
| Yaw tolerance | **35°** | **30°** | **25°** |
| Drone start | [4, 0, 6] | [5, 0, 6] | [8, -3, 7] |
| Duration | 60 s | 60 s | 60 s |

HARD's yaw tolerance is 25° (not 20° as the README implied). Some slack — but still want active yaw for repeatability.

**Yaw alignment is checked in boat body frame** (env.py:867-875), mod π. Both `ψ_drone == ψ_boat` AND `ψ_drone == ψ_boat + π` count as aligned ("wings perpendicular to boat length" or "wings along boat length" — both are landing-valid).

---

## Baseline (what's there)

Pure thresholding: SEARCH → APPROACH → DESCEND → LAND. No hysteresis. No abort. No battery awareness. Returns a string. LAND is bang-bang (control reads phase=="LAND" and pins thrust=-1).

---

## Ranked work, in order

### 1. Hysteresis + frame debouncing — ~1 h, biggest unlock
Asymmetric thresholds (entry tighter than exit) + N-frame debounce.

| Transition | Enter | Exit (abort back) |
|---|---|---|
| SEARCH → APPROACH | `fresh` AND `horiz < 0.6` | `horiz > 1.2` OR marker lost ≥8 frames |
| APPROACH → DESCEND | `horiz < 0.4` for ≥5 frames AND `z_above < 3.5` | `horiz > 0.8` OR marker lost ≥8 frames |
| DESCEND → LAND | `horiz < 0.25` AND `z_above < 0.6` | `horiz > 0.5` OR attitude tilt > 25° |

Internal frame counters per condition. The N-frame debounce matters more than the exact threshold values.

### 2. Two-stage approach gating (new from research) — ~30 min
Before DESCEND, hold at an intermediate altitude (`h_hold ≈ 1.5 m`) until BOTH:
- `‖horiz_error‖ < 0.3 m`
- `‖boat_relative_velocity‖ < 0.3 m/s`

This is the **MBZIRC standard pattern** for landing on a moving target ([Fast autonomous landing IROS 2017](https://ieeexplore.ieee.org/abstract/document/8098669/)). Without it, you commit to descent while still drifting and miss the platform.

```python
if phase == "APPROACH" and horiz_ok and not yet stable enough:
    # tell control to hold at h_hold, don't descend further
    target_descent_rate = 0
elif horiz_stable and v_rel_stable:
    transition to YAW_ALIGN or DESCEND
```

### 3. Return a rich dict, not a string — ~1 h
Per `contracts.md`:
```python
return {
    "phase": "DESCEND",
    "target_descent_rate": 0.6,    # m/s — control uses this
    "yaw_locked": True,             # control enables active yaw
    "commit_soft": True,            # control uses soft profile vs bang-bang
    "abort_reason": None,
}
```

Descent rate ramp:
- SEARCH / APPROACH / YAW_ALIGN: 0
- DESCEND: 0.6 m/s (down from draft-1's 0.6 — research said lower v_max ≈ 0.6 was better)
- LAND: 0.25 m/s if `commit_soft` else 0.5 m/s

Control reads these via `.get()` and falls back to baseline behavior when missing.

### 4. Time-and-battery-aware commit — ~1.5 h
Two scalars: `time_budget = 60 - time_s`, `battery_budget = battery`.

Rules:
- `time_budget < 15 s` AND `phase == SEARCH`: enter **commit mode**. Accept any detection (skip the 3-frame trust requirement). Skip YAW_ALIGN. Move directly to APPROACH.
- `battery_budget < 0.25`: bias toward DESCEND. Lower the APPROACH→DESCEND altitude bar to 4.5 m.
- After `time_s > 50` OR `z_above < 1.5 m`: **disable abort-to-APPROACH** (commit-to-land — better to crash near the platform than time out).

Expected-value heuristic:
```python
ev = precision_factor_estimate * time_factor_now
if ev < 0.4 and time_budget < 20:
    force_commit = True
```

### 5. Cold-start search — ~1 h
Replace static `(0, 0, 0.3)` prior with ascending spiral:
```python
center = SEARCH_PRIOR_XY
radius = min(0.5 + 0.5 * t, 6.0)
target_xy = center + radius * np.array([np.cos(ω*t), np.sin(ω*t)])
target_z = min(4.0 + 0.5 * t, 6.0)  # cap on HARD: fog density 0.10 attenuates at altitude
```
Higher altitude = wider FOV = faster acquisition.

### 6. YAW_ALIGN sub-phase — ~45 min
Insert between APPROACH and DESCEND.

**Enter** when:
- `phase == APPROACH`
- `horiz < 0.4`
- `boat_estimate.heading is not None`
- heading variance < (10°)² over a 1-s window

**Exit** (to DESCEND) when: `|yaw_err| < 5°` for ≥10 frames OR `time_in_phase > 4 s`.

**Skip entirely** if `time_budget < 20 s` — yaw is mod π anyway; worst-case yaw misalignment costs ~0 on precision but saves seconds.

### 7. Soft-landing gate — ~30 min
Set `commit_soft = True` only when **all** of:
- `‖v_rel_horiz‖ < 0.3 m/s`
- `|v_rel_vert| < 0.4 m/s`
- marker tracked ≥10 frames
- `z_above < 1.0 m`

Otherwise `commit_soft = False` (bang-bang). The +5 bonus isn't worth missing the platform.

### 8. Yaw lock-and-freeze (new from research) — ~15 min
Once `yaw_locked == True` is set, **freeze the yaw target** in `decide()`'s memory — don't keep tracking the boat heading as it (slightly) drifts during descent. The web research [agile UAV ship landing arXiv 2411.05445](https://arxiv.org/pdf/2411.05445) flagged continuous yaw tracking during descent as causing roll/pitch authority bleed on weak-yaw airframes (which the VTOL is — `k_Q/k_T = 0.020`, `Izz = 5.74`).

```python
if not self._yaw_target_locked and conditions_met:
    self._frozen_yaw_target = boat_estimate["heading"]
    self._yaw_target_locked = True
# control reads phase_out["yaw_locked"] = self._yaw_target_locked
# and uses self._frozen_yaw_target instead of live boat_estimate.heading
```

(Implementation note: `decide` keeps the frozen target; pass it via `phase_out` as `"yaw_target": float` if you don't want shared state across `decide`/`control`. Both options are fine.)

---

## What to skip

- Sophisticated MPC for the decision layer (FSM is fine here)
- Continuous "phase score" instead of discrete phases (debugging cost)
- Multi-step lookahead (no real planning needed)
- Battery state-of-charge models beyond linear

---

## Acceptance checklist

- [ ] Hysteresis + frame debouncing on every transition
- [ ] Two-stage approach gating (h_hold then descend)
- [ ] Returns dict (or string in baseline fallback path), control reads `.get()` defensively
- [ ] At least one abort path implemented (recommend: marker-lost during DESCEND → APPROACH)
- [ ] Time-budget commit at `time_s > 45 s` (commit-to-land)
- [ ] Search spiral implemented (or accept the static prior with explicit comment)
- [ ] YAW_ALIGN reachable when `boat_estimate.heading` is populated
- [ ] Yaw target freezes once locked (no tracking during DESCEND/LAND)
- [ ] Latency: state-machine evaluation is microseconds; not a budget concern

**Total budget:** ~3-4 hours for everything above hysteresis. Hysteresis + two-stage gating are the highest-leverage ~1.5 h.
