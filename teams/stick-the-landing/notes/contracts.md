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

Exposes the boat estimate for the scoring bonus (0–15 pts via RMSE). Keep returning `velocity` populated — that's table-stake under the new schema.

---

## Merge rules

1. **No PR adds a required field** without team agreement. Optional + default is always OK.
2. **Renames are a refactor PR** — coordinate first.
3. **Readers must `.get()`**, never key-access optional fields directly.
4. Type-annotate signatures with the dict shapes inline (a `TypedDict` if you feel like it, but optional). Comments are enough.
5. Each stage owner's PR should leave the baseline behavior intact for fields they don't use.
