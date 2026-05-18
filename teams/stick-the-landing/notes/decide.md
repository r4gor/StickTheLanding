# Decision-layer playbook

**Owner suggestion:** one of the CS devs (probably the one who also owns the eval harness — there's overlap on understanding the scoring formula).

**Goal:** eliminate phase flicker, add abort/retry, prepare for the soft-landing bonus. Single biggest leverage: **hysteresis + frame debouncing typically moves median score 15-30%** by killing flicker-induced failures.

---

## Baseline (what's there)

Pure thresholding: SEARCH → APPROACH → DESCEND → LAND. No hysteresis. No abort. No battery awareness. Returns a string. LAND is bang-bang (control reads phase=="LAND" and pins thrust=-1).

---

## Ranked work, in order

### 1. Hysteresis + frame debouncing — ~1 h, biggest unlock
Asymmetric thresholds (entry tighter than exit) + N-frame debounce.

```
# Frame-debounced "approach-tight" condition: horiz < 0.4 for ≥5 frames
# Frame-debounced "descend-stable" condition: horiz < 0.25 for ≥10 frames
```

| Transition | Enter | Exit (abort back) |
|---|---|---|
| SEARCH → APPROACH | `fresh` AND `horiz < 0.6` | `horiz > 1.2` OR marker lost ≥8 frames |
| APPROACH → DESCEND | `horiz < 0.4` for ≥5 frames AND `z_above < 3.5` | `horiz > 0.8` OR marker lost ≥8 frames |
| DESCEND → LAND | `horiz < 0.25` AND `z_above < 0.6` | `horiz > 0.5` OR attitude tilt > 25° |

Internal frame counters per condition. The N-frame debounce matters more than the exact threshold values.

### 2. Return a rich dict, not a string — ~1 h
Per `contracts.md`:
```python
return {
    "phase": "DESCEND",
    "target_descent_rate": 0.6,   # m/s — control uses this
    "yaw_locked": True,            # control enables active yaw
    "commit_soft": True,           # control uses soft profile vs bang-bang
    "abort_reason": None,
}
```

Descent rate ramp:
- SEARCH: 0
- APPROACH: 0
- YAW_ALIGN: 0
- DESCEND: 0.6 m/s
- LAND: 0.25 m/s if `commit_soft` else 0.5 m/s

**Control reads these fields and falls back to baseline behavior when missing** — so this can ship independently.

### 3. Time-and-battery-aware commit — ~1.5 h
Two scalars: `time_budget = 60 - time_s`, `battery_budget = battery`.

Rules:
- `time_budget < 15 s` AND `phase == SEARCH`: enter **commit mode**. Accept any detection (skip the 3-frame trust requirement). Skip YAW_ALIGN. Move directly to APPROACH.
- `battery_budget < 0.25`: bias toward DESCEND. Lower the APPROACH→DESCEND altitude bar to 4.5 m.
- After `time_s > 50` OR `z_above < 1.5 m`: **disable abort-to-APPROACH** (commit-to-land — better to crash near the platform than time out).

Expected-value heuristic if you have time:
```python
ev = precision_factor_estimate * time_factor_now
if ev < 0.4 and time_budget < 20:
    force_commit = True
```

### 4. Cold-start search — ~1 h
Replace static `(0, 0, 0.3)` prior with ascending spiral:
```python
# Starting from last known prior (or origin if none):
center = SEARCH_PRIOR_XY
radius = min(0.5 + 0.5 * t, 6.0)         # grow 0.5 m/s, cap 6 m
target_xy = center + radius * np.array([np.cos(ω*t), np.sin(ω*t)])
target_z = min(4.0 + 0.5 * t, 8.0)       # climb to widen FOV
# (cap target_z at 6 m on HARD because of fog)
```
Higher altitude = wider FOV = faster acquisition.

### 5. YAW_ALIGN sub-phase — ~45 min
Insert between APPROACH and DESCEND.

**Enter** when: `phase == APPROACH` AND `horiz < 0.4` AND `boat_estimate.heading is not None` AND `heading` variance < (10°)² over a 1-s window.

**Exit** (to DESCEND) when: `|yaw_err| < 5°` for ≥10 frames OR `time_in_phase > 4 s`.

Skip entirely if `time_budget < 20 s` — yaw is mod π anyway, worst case costs ~0 on precision but saves seconds.

### 6. Soft-landing gate — ~30 min
Set `commit_soft = True` only when **all** of:
- `‖v_rel_horiz‖ < 0.3 m/s`
- `|v_rel_vert| < 0.4 m/s`
- marker tracked ≥10 frames
- `z_above < 1.0 m`

Otherwise `commit_soft = False` (bang-bang). The bonus isn't worth missing the platform.

---

## What to skip

- Sophisticated MPC for the decision layer (FSM is fine here)
- Continuous "phase score" instead of discrete phases (debugging cost)
- Multi-step lookahead (no real planning needed)
- Battery-state-of-charge models beyond linear (you don't need accuracy here)

---

## Acceptance checklist

- [ ] Hysteresis + frame debouncing on every transition
- [ ] Returns dict (or string in baseline fallback path), control reads `.get()` defensively
- [ ] At least one abort path implemented (recommend: marker-lost during DESCEND → APPROACH)
- [ ] Time-budget commit at `time_s > 45 s` (commit-to-land)
- [ ] Search spiral implemented (or accept the static prior with explicit comment)
- [ ] YAW_ALIGN reachable when `boat_estimate.heading` is populated
- [ ] Latency: state-machine evaluation is microseconds; not a budget concern

**Total budget:** ~3-4 hours for everything above hysteresis. Hysteresis alone is ~1 h and worth the most.
