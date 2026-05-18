# Agent stage I/O contracts (locked before fork)

Each stage in `agent.py` is a method that takes a dict and returns a dict. To make everyone's edits **additively mergeable**, we pin the schemas here. **New optional fields are allowed**; renames/removes are not.

Readers MUST use `.get(key, default)` for everything marked **optional** so old code keeps working as new fields land.

---

## `perceive(camera) → percept`

```python
percept = {
    # required
    "detected":            bool,
    "tvec":                np.ndarray (3,) | None,   # camera frame [m]
    "rvec":                np.ndarray (3,) | None,   # camera-frame Rodrigues
    "image_points":        np.ndarray (4,2) | None,  # pixel corners

    # optional (added during the perception upgrade)
    "reprojection_error":  float | None,             # pixels, mean per-corner
    "confidence":          float | None,             # 0..1 derived from reproj_error
    "ambiguous":           bool | None,              # IPPE_SQUARE two solutions too close
}
```

**Stateless.** No drone state in, no history. Pure CV.

---

## `estimate(percept, drone_state) → boat_estimate`

> **Change:** drop the `history=None` parameter — it's dead in the baseline. Filter state lives on `self`.

```python
boat_estimate = {
    # required (kept compatible with baseline)
    "position":      np.ndarray (3,),   # world frame [m]
    "velocity":      np.ndarray (3,),   # world frame [m/s] — KF must populate
    "fresh":         bool,
    "stale_steps":   int,
    "from_prior":    bool,

    # optional (new)
    "heading":       float | None,      # boat yaw [rad], mod π for landing condition
    "heading_rate":  float | None,      # [rad/s]
    "position_cov":  np.ndarray (3,3) | None,  # KF covariance — lets decide() abort
    "velocity_cov":  np.ndarray (3,3) | None,
    "confidence":    float | None,      # 0..1 scalar; KF derives from trace(P_pos). Lets decide() write `if confidence > 0.7: commit_soft = True` without inverting matrices.
}
```

**Stateful** on `self` (KF instance variables). New fields stay `None` until estimate stage owner ships them; control/decide read with `.get()`.

---

## `decide(drone_state, boat_estimate, battery, time_s) → phase_out`

> **Change:** return a dict, not a string. Lets `decide()` pass tuning knobs to `control()` without growing argument count.

```python
phase_out = {
    # required
    "phase":               str,    # SEARCH | APPROACH | YAW_ALIGN | DESCEND | LAND
    
    # optional (new)
    "target_descent_rate": float | None,   # m/s, used by control's soft-landing
    "yaw_locked":          bool | None,    # True if heading estimate is trustworthy
    "commit_soft":         bool | None,    # if True, control uses soft profile instead of bang-bang
    "abort_reason":        str | None,     # human-readable, for logging only
}
```

**Backwards-compat helper:** if `phase_out` ends up being a bare string in some path, control should accept `isinstance(phase_out, str)` and fall back to defaults. Or wrap once: `phase = phase_out if isinstance(phase_out, str) else phase_out["phase"]`. This keeps the baseline working while improvements ramp.

**New phase:** `YAW_ALIGN` slots between `APPROACH` and `DESCEND`. Control should treat it like APPROACH for position but actively yaw.

---

## `control(drone_state, boat_estimate, phase_out) → motor_throttles`

Same signature, same output `(N,) ∈ [0,1]`. Internally, control reads:
- `phase_out.get("target_descent_rate")` for soft-landing
- `phase_out.get("commit_soft")` to decide bang-bang vs profile
- `phase_out.get("yaw_locked")` to enable active yaw
- `boat_estimate.get("velocity", zeros(3))` for feedforward
- `boat_estimate.get("heading")` for yaw error

---

## `get_last_estimate() → {position, velocity}`

Exposes the boat estimate for the scoring bonus (0–15 pts). Per `scorer.py:37-44`, the formula is `15 * (1 - min(rmse, 2.0) / 2.0)` using **2D XY position only** — z and velocity are in the return dict for convention but not scored. Keep returning `velocity` anyway because `control()` reads it.

Also: this method must keep returning a valid estimate during the **recovery test** (camera blacked out by `occlusion_window`). KF predict-only handles that — just don't return `None` while occluded.

---

## `act_setpoint(obs) → (thrust_norm, roll, pitch, yaw_rate)` ← NEW, earns +15 pts FC-compat

`evaluate.py:146-163` checks for this method **by name**. Its presence alone earns the FC-compatibility bonus (no format validation on the output). Tuple shape is exactly `(thrust_norm: float, roll: float, pitch: float, yaw_rate: float)` — the four high-level setpoints the cascade already computes.

Recommended implementation: refactor `control()` into a private `_control_setpoints()` that returns the 4-tuple, then have both `control()` and `act_setpoint()` call it:

```python
def _control_setpoints(self, state, boat_estimate, phase_out):
    # ... existing control logic ...
    return {"thrust": thrust_cmd, "roll": roll_cmd,
            "pitch": pitch_cmd, "yaw_rate": yaw_rate_cmd}

def control(self, state, boat_estimate, phase_out):
    sp = self._control_setpoints(state, boat_estimate, phase_out)
    return self.attitude_ctrl(state, sp["roll"], sp["pitch"],
                              sp["yaw_rate"], sp["thrust"])

def act_setpoint(self, obs):
    p  = self.perceive(obs["camera"])
    be = self.estimate(p, obs["state"])
    po = self.decide(obs["state"], be, obs["battery"], obs["time"])
    sp = self._control_setpoints(obs["state"], be, po)
    return (sp["thrust"], sp["roll"], sp["pitch"], sp["yaw_rate"])
```

Additive: `control()` signature unchanged, baseline `act()` unchanged. The control-stage owner does this refactor; nobody else has to coordinate.

---

## NaN / Inf sanitization (mandatory)

`scorer.py:128-143` **forces non-finite inputs to worst-case defaults** (NaN landing-error → 1.5 m → zero precision; NaN battery → 0 → 0.3 floor; NaN descent-vel → ∞ → no soft-landing bonus). Emitting NaN is therefore worse than emitting `None`. **Every numeric field above must be either finite or `None`** — never NaN/Inf.

Concretely:
- `perceive()` must return `"detected": False` (not `True` with NaN tvec). Baseline already guards this — keep it.
- `estimate()` must guard against NaN propagation (e.g., NaN drone yaw → world transform NaN).
- `_control_setpoints()` clamps every output to a finite range before returning.
- Baseline has a hover fallback in `act()` (`if not np.all(np.isfinite(action))`) — keep it.

---

## Merge rules

1. **No PR adds a required field** without team agreement. Optional + default is always OK.
2. **Renames are a refactor PR** — coordinate first.
3. **Readers must `.get()`**, never key-access optional fields directly.
4. Type-annotate signatures with the dict shapes inline (a `TypedDict` if you feel like it, but optional). Comments are enough.
5. Each stage owner's PR should leave the baseline behavior intact for fields they don't use.
