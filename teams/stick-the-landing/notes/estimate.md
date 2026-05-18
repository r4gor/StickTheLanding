# Estimation stage playbook

**Owner suggestion:** the AI/ML dev. **The single biggest agent-track unlock** per the docs.

**Goal:** populate `boat_estimate.velocity` (currently zeros — bleeds the estimation bonus) and produce robust boat-state under noisy / missing detections at 10-50 Hz.

---

## Recommended design

**Decouple translation from heading.**

- **4-state linear Kalman filter** on `[x, y, vx, vy]` (world frame) — handles position + velocity.
- **Separate scalar Kalman filter** on `[ψ, ψ̇]` (boat heading) — from `rvec`, mod π.

Why this and not the alternatives:

| Design | Verdict | Reason |
|---|---|---|
| CV 4-state KF (recommended) | ✅ | Linear, fast, debuggable. Boat acceleration absorbed by Q. |
| CA 6-state | skip | Adds states for marginal gain. CV with right Q wins. |
| SE(2) full pose EKF/UKF | skip | Nonlinear coupling, debugging nightmare in 12h. Heading is decoupled at this scale. |
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
body_z = R_world_body[:, 2]
p_camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z   # -0.115 m
R_cam_to_world = column_stack([-body_y, -body_x, -body_z])    # from agent_pipeline.md
marker_world = p_camera_world + R_cam_to_world @ tvec
z_k = marker_world[:2]
```

---

## Tuning (concrete numbers)

### R — measurement noise (2×2)
Scale with altitude and fog/confidence:
```
σ_xy_base = 0.05                            # m, 1σ at nominal altitude
σ_xy = σ_xy_base * (drone_alt / 5.0) ** 1.0   # PnP error grows ~linearly with range
σ_xy *= max(1.0, reprojection_error / 1.0)   # inflate if reproj_error > 1 px

# Or simpler closed form:
# σ_xy ≈ k · reproj_rms_px · range / focal_length_px
R = np.diag([σ_xy**2, σ_xy**2])
```

On HARD, expect `σ_xy ≈ 0.15` m baseline. If `percept["ambiguous"]`, inflate R by 9× (3× σ).

### Q — process noise (4×4, discretized CV white-noise-acceleration)
```
σ_a = 1.0   # m/s² for EASY/MEDIUM (boat is slow, oscillation absorbed)
σ_a = 3.0   # m/s² for HARD (curve at 2.5 m/s ≈ 1.5 m/s² centripetal + gusts)
Q = σ_a**2 * np.array([
    [dt**4/4, 0,       dt**3/2, 0],
    [0,       dt**4/4, 0,       dt**3/2],
    [dt**3/2, 0,       dt**2,   0],
    [0,       dt**3/2, 0,       dt**2],
])
```
You don't know the scenario at runtime — use a single conservative `σ_a = 2.0` m/s² or detect HARD via the camera fps.

---

## Missing-measurement handling

When `detected == False`:
- **Predict-only step**: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
- Increment `stale_steps`.
- After `stale_steps · dt > 1.5 s`: double P entries every additional second. Mark `fresh=False`.
- After `stale_steps · dt > 5 s`: **reset** — zero velocity, set position cov to `diag(4, 4)` m², return `from_prior=True`. Control should fall back to search.

---

## Initialization

- First detection: seed `x = [x_meas, y_meas, 0, 0]`, `P = diag(σ_xy², σ_xy², 1, 1)`.
- Mark filter "trusted" (i.e., `fresh=True`) only after **3 consecutive detections** with NIS < 9.0.
- Until trusted, `velocity = 0`. After trusted, return the filtered velocity.

---

## Outlier rejection (HARD-relevant)

Mahalanobis / NIS gating on every measurement:
```
y = z - H @ x_pred                  # innovation
S = H @ P_pred @ H.T + R            # innovation cov (2x2)
NIS = y.T @ np.linalg.inv(S) @ y
if NIS > 9.21:                      # 2-DoF, 99%
    skip update; predict-only
    consecutive_rejects += 1
if consecutive_rejects > 5:
    reinitialize from latest measurement
else:
    consecutive_rejects = 0
```
Worth it on HARD. Optional on EASY/MEDIUM (low false-rate to begin with).

---

## Heading filter (scalar KF on [ψ, ψ̇])

**Measurement source — rvec primarily:**
```
R_marker_cam, _ = cv2.Rodrigues(rvec)
R_marker_world = R_world_cam @ R_marker_cam
ψ_meas = np.arctan2(R_marker_world[1, 0], R_marker_world[0, 0])
# Landing condition is mod π, so wrap:
ψ_meas = ((ψ_meas + np.pi/2) % np.pi) - np.pi/2
```

**Tuning:** σ_ψ_meas ≈ 0.05 rad, σ_ψ̇_process ≈ 0.3 rad/s. Same KF structure, scalar.

**Velocity-direction fallback** (`atan2(vy, vx)`) only when `‖v‖ > 1.0 m/s` AND no detection for >1 s. Don't use it on EASY (static boat) or during oscillation.

---

## Outputs (per `contracts.md`)

```python
return {
    "position":      np.array([x_kf, y_kf, last_z_meas]),
    "velocity":      np.array([vx_kf, vy_kf, 0.0]),
    "fresh":         True if trusted else False,
    "stale_steps":   int,
    "from_prior":    bool,
    "heading":       ψ_kf,             # or None until lock
    "heading_rate":  ψ̇_kf,
    "position_cov":  P_kf_pos,         # 3x3 (vz row/col = inf or large)
    "velocity_cov":  P_kf_vel,
}
```

---

## Leverage ranking (in order to ship)

1. **Velocity output populated** (kills the biggest part of the bonus deficit)
2. **R scaling with altitude/reprojection_error** (huge on HARD)
3. **rvec-based heading filter** (unlocks active yaw in control)
4. **Predict-only + cov inflation for missing measurements** (HARD survival)
5. **NIS gating** (HARD only)
6. **Divergence reset on 5 s dropout**

## Acceptance checklist

- [ ] KF state + covariance maintained on `self`
- [ ] All `boat_estimate` optional fields populated when trusted
- [ ] `get_last_estimate()` returns the KF position + velocity (not raw last-marker)
- [ ] Unit test: synthetic constant-velocity boat at 0.5 m/s + noisy measurements → velocity converges to ±10% within 2 s
- [ ] Latency: KF update should be << 1 ms; well within 20 ms budget
