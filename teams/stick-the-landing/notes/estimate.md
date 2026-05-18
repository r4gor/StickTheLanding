# Estimation stage playbook

**Owner suggestion:** the AI/ML dev. **The single biggest agent-track unlock** per the docs.

**Goal:** populate `boat_estimate.velocity` (currently zeros — bleeds the estimation bonus) and produce robust boat-state under noisy / missing detections at 15-50 Hz.

> **Research validation (2026-05-18):** The 4-state CV KF design is canonical — PX4 ships exactly this state vector and dynamics in `landing_target_estimator/LandingTargetEstimator.cpp`. ArduPilot's `AC_PrecLand` ships a "mini-EKF" variant that drops landing error from 30-50 cm to ~5 cm vs raw mode ([ArduPilot discourse](https://discuss.ardupilot.org/t/irlock-raw-data-or-kalman-filter/23398)). **Production stacks do NOT estimate marker heading from `rvec`** — they expect autopilot yaw from GPS/compass. Estimating ψ from `rvec` is *your* edge for the bonus.

---

## Scoring math reality check

Per `scorer.py:37-44`:
```
estimation_bonus = 15 * (1 - min(rmse, 2.0) / 2.0)     # linear from 0m=15pts to 2m=0pts
rmse = sqrt(mean((true_xy - estimated_xy)²))            # 2D XY position only (z and velocity ignored)
```

So the bonus is **purely on 2D XY position RMSE**. Velocity and z are not scored — but you still populate them for `control()`'s feedforward and for the recovery bonus.

Recovery bonus (also +15) requires `get_last_estimate()` to keep returning a sensible value during the camera blackout window, and the position must agree with truth within 0.3 m within 1 s after the blackout ends. **KF predict-only handles this for free.**

---

## Recommended design

**Decouple translation from heading.**

- **4-state linear Kalman filter** on `[x, y, vx, vy]` (world frame) — handles position + velocity.
- **Separate scalar Kalman filter** on `[ψ, ψ̇]` (boat heading) — from `rvec`, mod π.

Why this and not the alternatives:

| Design | Verdict | Reason |
|---|---|---|
| CV 4-state KF (recommended) | ✅ | Linear, fast, debuggable. Matches PX4 production code. |
| CA 6-state | skip | Boat lateral accel on HARD is ~0.11 m/s² (v²/r = 1.8²/30) — CV with right Q absorbs it. |
| SE(2) full pose EKF/UKF | skip | Nonlinear coupling, debugging nightmare in 12h. |
| Particle filter | skip | Overkill — measurement is unimodal Gaussian after PnP. |
| Alpha-beta filter | skip | Loses covariance (needed for gating + abort logic) |

---

## State + measurement model

```
x_k = [x, y, vx, vy]ᵀ
F   = [[1, 0, dt, 0],
       [0, 1, 0, dt],
       [0, 0, 1,  0],
       [0, 0, 0,  1]]
H   = [[1, 0, 0, 0],
       [0, 1, 0, 0]]
z_k = [x_meas, y_meas]ᵀ   ← world-frame marker position from perception
```

**Measurement construction (world frame):**
```
R_world_body = rpy_to_matrix(drone_state["attitude"])
body_z       = R_world_body[:, 2]
p_cam_world  = drone_pos + body_z * CAMERA_BODY_OFFSET_Z   # -0.115 m
R_cam_world  = column_stack([-body_y, -body_x, -body_z])    # see agent_pipeline.md
marker_world = p_cam_world + R_cam_world @ tvec
z_k = marker_world[:2]
```

**Free RMSE win** ([UZH RPG Falanga 2017](https://rpg.ifi.uzh.ch/docs/SSRR17_Falanga.pdf)): subtract `drone_state["velocity"][:2]` when treating the measurement noise model — the marker measurement is *relative* (camera moves with drone). The KF tracks world-frame boat motion at lower Q if you compensate.

---

## Tuning (concrete numbers, **updated for actual scenario params**)

### R — measurement noise (2×2)
Scale with altitude and reprojection error:
```
σ_xy_base = 0.05                            # m, 1σ at nominal altitude
σ_xy = σ_xy_base * (drone_alt / 5.0)        # PnP error grows ~linearly with range
σ_xy *= max(1.0, percept["reprojection_error"] / 1.0)
R = np.diag([σ_xy**2, σ_xy**2])
```

On HARD, expect `σ_xy ≈ 0.10` m baseline (less than draft-1's 0.15 because fog density is 0.10, not 0.15). If `percept["ambiguous"]`, inflate R by 9× (3× σ).

### Q — process noise (4×4, discretized CV white-noise-acceleration)
**Corrected:** HARD is gentler than originally assumed. Actual params: 1.8 m/s curved at r=30 m → centripetal accel ≈ 0.11 m/s². Add gust accel from wind (0.5 N gust / 10 kg = 0.05 m/s²) and oscillation. **σ_a ≈ 2 m/s² is the right ballpark for HARD** (1.0 for EASY/MEDIUM).
```
σ_a = 1.0 if scenario_easy_or_medium else 2.0
Q = σ_a**2 * np.array([
    [dt**4/4, 0,       dt**3/2, 0],
    [0,       dt**4/4, 0,       dt**3/2],
    [dt**3/2, 0,       dt**2,   0],
    [0,       dt**3/2, 0,       dt**2],
])
```
You don't know the scenario at runtime — use a single conservative `σ_a = 2.0` m/s² across the board. The maneuvering-target literature confirms CV with appropriate Q absorbs brief maneuvers but is slightly biased during sustained turns ([NAVSWC TR 91-701](https://apps.dtic.mil/sti/tr/pdf/ADA254949.pdf)). For this challenge, that bias costs <0.1 m on RMSE — well below the bonus saturation.

**If you have time:** swap WNA for Singer (τ=15s, σ=2 m/s²) to model correlated accelerations during boat turns ([MathWorks singerProcessNoise](https://www.mathworks.com/help/fusion/ref/singerprocessnoise.singerprocessnoise.html)). ~50 lines drop-in. Skip unless RMSE > 0.5 m on HARD in dry-run.

---

## Missing-measurement handling (= recovery bonus path)

When `detected == False`:
- **Predict-only step**: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
- Increment `stale_steps`.
- After `stale_steps · dt > 1.5 s`: double P entries every additional second. Mark `fresh=False`.
- After `stale_steps · dt > 5 s`: **reset** — zero velocity, set position cov to `diag(4, 4)` m², return `from_prior=True`. Control should fall back to search.

**During the recovery test** (`evaluate.py:166-228`), the camera is blacked out for `occlusion_duration` seconds. KF predict-only is exactly what gets you through: the drone sees no marker, predicts the boat forward at `(vx, vy)`, and when the blackout ends, the first detection re-locks. Recovery passes if the drone stayed within 2 m horizontally AND the KF estimate is within 0.3 m of truth within 1 s after blackout.

---

## Initialization

- First detection: seed `x = [x_meas, y_meas, 0, 0]`, `P = diag(σ_xy², σ_xy², 1, 1)`.
- Mark filter "trusted" (i.e., `fresh=True`) only after **3 consecutive detections** with NIS < 9.0.
- Until trusted, `velocity = 0`. After trusted, return the filtered velocity.

---

## Outlier rejection (HARD-relevant)

Chi-squared NIS gating. **Use threshold 13.8 (99.9%), not 9.21**. Recent result: gated NIS is *not* chi-square distributed under heavy gating (5% occlusion + fog), so tuning to χ²(2,0.99) under-estimates true innovation variance ([arXiv 2512.18508](https://arxiv.org/abs/2512.18508)).
```
y = z - H @ x_pred                  # innovation
S = H @ P_pred @ H.T + R            # innovation cov (2x2)
NIS = y.T @ np.linalg.inv(S) @ y
if NIS > 13.8:                      # 99.9% (was 9.21 in draft-1)
    skip update; predict-only
    consecutive_rejects += 1
if consecutive_rejects > 5:
    reinitialize from latest measurement
else:
    consecutive_rejects = 0
```
Worth it on HARD. Optional on EASY/MEDIUM.

**Reset-on-flip detector (new from research):** if `percept["ambiguous"]` AND new `rvec`-derived heading differs by >90° from current filtered heading, **reject the measurement entirely** — don't pass through with low confidence. This catches the IPPE flip before it pollutes the heading KF.

---

## Heading filter (scalar KF on [ψ, ψ̇])

**Measurement source — rvec, but GATED:**
```
R_marker_cam, _ = cv2.Rodrigues(rvec)
R_marker_world  = R_world_cam @ R_marker_cam
ψ_meas = np.arctan2(R_marker_world[1, 0], R_marker_world[0, 0])
ψ_meas = ((ψ_meas + np.pi/2) % np.pi) - np.pi/2     # mod π wrap
```

**Gate ψ̇ update on:** `confidence > 0.5 AND tilt > 15°`. At low tilt the IPPE flip pollutes heading directly; published yaw error is 6.6° baseline vs 3.1° for learned regressors ([PMC12196723](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12196723/)).

**Tuning:** σ_ψ_meas ≈ 0.05 rad, σ_ψ̇_process ≈ 0.3 rad/s. Same KF structure, scalar.

**Velocity-direction fallback** (`atan2(vy, vx)`) only when `‖v‖ > 1.0 m/s` AND no detection for >1 s. Don't use it on EASY (static boat).

---

## Outputs (per `contracts.md`)

```python
return {
    "position":      np.array([x_kf, y_kf, last_z_meas]),
    "velocity":      np.array([vx_kf, vy_kf, 0.0]),
    "fresh":         True if trusted else False,
    "stale_steps":   int,
    "from_prior":    bool,
    "heading":       ψ_kf,                          # or None until lock
    "heading_rate":  ψ̇_kf,
    "position_cov":  P_kf_pos,                      # 3×3 (vz row/col large)
    "velocity_cov":  P_kf_vel,
    "confidence":    1.0 / (1.0 + np.trace(P_kf_pos[:2,:2])),   # 0..1 scalar
}
```

`get_last_estimate()` returns `{position: np.array([x_kf, y_kf, last_z_meas]), velocity: np.array([vx_kf, vy_kf, 0])}`. Must not return `None` during occlusion (predict-only keeps it valid).

---

## Leverage ranking (in order to ship)

1. **Velocity output populated** (kills the biggest part of the bonus deficit)
2. **R scaling with altitude/reprojection_error** (huge on HARD)
3. **Predict-only during occlusion** (unlocks recovery bonus +15 if available)
4. **rvec-based heading filter** (unlocks active yaw in control)
5. **Reset-on-flip detector** (HARD safety)
6. **NIS gating at threshold 13.8** (HARD only)
7. **Divergence reset on 5 s dropout**

---

## Acceptance checklist

- [ ] KF state + covariance maintained on `self`
- [ ] All `boat_estimate` optional fields populated when trusted (including `confidence`)
- [ ] `get_last_estimate()` returns the KF position + velocity (not raw last-marker)
- [ ] `get_last_estimate()` keeps working during a 1-second simulated camera blackout
- [ ] No NaN ever propagates into outputs (guard drone-state transforms)
- [ ] Unit test: synthetic constant-velocity boat at 1.0 m/s + noisy measurements → velocity converges to ±10% within 2 s
- [ ] Unit test: simulated 1-s camera blackout → estimate stays within 0.3 m of truth
- [ ] Latency: KF update << 1 ms; well within 20 ms budget
