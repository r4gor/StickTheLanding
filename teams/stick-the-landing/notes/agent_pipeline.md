# Agent pipeline — perceive → estimate → decide → control

Mathy reference for `agents/agent_baseline.py` (453 LOC). The class is intentionally split into four overridable stages; each is a clean hook point for improvements.

```
obs → perceive() → estimate() → decide() → control() → motor throttles
       │            │            │           │
       ArUco+PnP    Cam→world    State       3 PIDs + AttitudeCtrl
                    + memory     machine
```

`act(obs)` just orchestrates the four. **Override stages, not `act`.**

---

## Stage 1 — `perceive(camera) → {detected, tvec, rvec, image_points}`

Pure CV. No drone state used.

| Step | Op | Output |
|---|---|---|
| 1 | `gray = cv2.cvtColor(img, RGB2GRAY)` | (H,W) uint8 |
| 2 | `corners, ids = ArucoDetector.detectMarkers(gray)` | list of (4,2) pixel corners + ids |
| 3 | filter to `id == 0` | one set of 4 corners |
| 4 | `cv2.solvePnP(obj_pts, img_pts, K, dist=0, IPPE_SQUARE)` | `rvec, tvec` in **camera frame** |
| 5 | reject NaN, `tvec[2] ≤ 0.1`, `tvec[2] > 100` | bool gate |

Model points (marker's local frame, z=0 plane, half-side `h = 0.4` m):
```
obj_pts = [(-h,+h,0), (+h,+h,0), (+h,-h,0), (-h,-h,0)]   # TL TR BR BL
```

`tvec ∈ ℝ³`: position of marker origin in the **camera** frame.

---

## Stage 2 — `estimate(perception, drone_state, history) → {position, velocity, fresh, stale_steps, from_prior}`

Transforms `tvec` into a **world-frame boat estimate**. Three branches: fresh detection, last known, cold-start prior.

### Fresh detection path

```
R_wb = Rz(yaw) · Ry(pitch) · Rx(roll)              # body→world, from drone.attitude
body_x, body_y, body_z = R_wb[:,0], R_wb[:,1], R_wb[:,2]

p_cam_world = p_drone + body_z · CAMERA_BODY_OFFSET_Z      # CAMERA_BODY_OFFSET_Z = −0.115 m

# Camera convention (OpenCV) vs body frame:
#   cam +X (image right) = −body_y
#   cam +Y (image down)  = −body_x
#   cam +Z (into scene)  = −body_z
R_cam_world = [−body_y | −body_x | −body_z]   (3×3, columns)

marker_world = p_cam_world + R_cam_world · tvec
```

Output: `position = marker_world`, `velocity = 0` (no derivative estimated — **baseline weakness #1**), `fresh = True`.

### Stale path
If no detection AND last marker known → return last marker, `velocity = 0`, `stale_steps++`, `fresh = False`.

### Cold-start prior
If never detected → return `(0, 0, 0.3)` as a guess. Hint: a real solution should *search*, not guess.

---

## Stage 3 — `decide(state, boat_estimate, battery, time) → phase ∈ {SEARCH, APPROACH, DESCEND, LAND}`

Pure thresholding on horizontal distance and altitude. **No hysteresis** — flicker is possible.

```
horiz   = ‖ p_drone[:2] − p_boat[:2] ‖
z_above =     p_drone[2]  − p_boat[2]

if z_above < 0.6                                       → LAND
elif horiz < 0.4 and z_above < 3.5                     → DESCEND
elif horiz < 0.6                                       → APPROACH
elif fresh detection                                   → APPROACH
else                                                   → SEARCH
```

`PHASE_ALTITUDE = {SEARCH: 5.5, APPROACH: 3.0, DESCEND: 0.4, LAND: −0.5}`. Each phase's target altitude is **strictly below** the next phase's transition threshold — otherwise the controller settles at equilibrium and never transitions.

---

## Stage 4 — `control(state, boat_estimate, phase) → motor_throttles ∈ [0,1]ᴺ`

Three independent PIDs + the stock attitude mixer.

### Reset on phase change
```
if phase != prev_phase:  pid_x.reset(), pid_y.reset(), pid_z.reset()
```

### Build errors in mixed frames

| Frame | Why |
|---|---|
| **Body** for xy errors | so pitch/roll commands work at any yaw |
| **World** for z error | altitude doesn't tilt with the drone |

```
target = p_boat
target_z = p_boat[2] + PHASE_ALTITUDE[phase]

R          = R_wb(rpy)
v_world    = drone_state.velocity
v_body     = Rᵀ · v_world

err_world  = (target − p_drone)                    # (3,)
err_body   = Rᵀ · err_world

pitch_cmd  =  PID_x( err_body[0],  dt,  v_meas = v_body[0]  )
roll_cmd   = −PID_y( err_body[1],  dt,  v_meas = v_body[1]  )
thrust_cmd =  PID_z( err_world[2], dt,  v_meas = v_world[2] )
yaw_rate_cmd = 0                                   # ← no active yaw (weakness #7)

if phase == LAND:  thrust_cmd = −1.0               # ← bang-bang (weakness #8)
```

### PID (per axis)
```
i  ← clip( i + err·dt,  ±i_clip )
d  ← −v_meas        if measurement_velocity given       ("D on measurement")
   |  (err − prev_err) / dt                              ("D on error")
out = clip( Kp·err + Ki·i + Kd·d,  ±1 )
```

D-on-measurement avoids "derivative kick" when the noisy boat estimate snaps to a fresh detection.

### Gains (deliberately conservative — headroom for teams)
| Axis | Kp | Ki | Kd |
|---|---|---|---|
| x | 0.15 | 0.00 | 0.40 |
| y | 0.15 | 0.00 | 0.40 |
| z | 0.10 | 0.02 | 0.30 |
| yaw | 1.0 | 0.0 | 0.0 (unused) |

### Mix to motors
```
motor_throttles = DefaultAttitudeController(state, roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd)
                                                    │
                                                    └─ stock; uses spec geometry + k_T/k_Q to invert
```

Output: `(N,) ∈ [0,1]` per-motor throttle.

### Optional: `get_last_estimate() → {position, velocity}`

Returning `(position, velocity)` of the boat estimate **unlocks the estimation bonus (0–15 pts)**, scored as RMSE against ground truth. Baseline returns `velocity = 0` so it eats most of the bonus on EASY but bleeds it on MEDIUM/HARD.

---

## Highest-leverage improvements (from the docstring's own "Deliberate weaknesses")

| # | Weakness | Fix | Expected gain |
|---|---|---|---|
| 1 | No boat motion prediction | **Kalman filter on (x, y, vx, vy)** (single biggest unlock per TIPS.md) | Lands MEDIUM/HARD; unlocks `velocity` for estimation bonus |
| 2 | Every detection trusted as-is | Kalman or robust smoother | Eliminates controller chasing noise |
| 3 | No marker-loss recovery | Predict-only KF step when `detected=False` | Handles HARD's fog/occlusion |
| 4 | Hardcoded thresholds + cold-start prior at origin | Proper search pattern (spiral / lawnmower) | Robustness when boat isn't near origin |
| 5 | No abort/retry on bad descent | Re-enter APPROACH if horiz drifts too far during DESCEND | Avoids "off the platform" failures |
| 6 | Independent x/y/z PIDs ignore tilt coupling | Couple xy via attitude or use MPC | Marginal vs the big two |
| 7 | **No active yaw control** | yaw-rate command from boat heading estimate (use marker's `rvec`) | Second-biggest unlock per docs (yaw-aligned landing condition) |
| 8 | LAND bang-bang descent | Velocity-feed-forward soft descent profile | **Stacks soft-landing bonus** (currently 0) |

**Two biggest unlocks:** #1 (Kalman) + #7 (yaw). Per TIPS.md, these are *the* leverage points named explicitly.

### HW-readiness bonuses (45 pts max, partly free)

| Bonus | Status | Path |
|---|---|---|
| FC-compatible output | ❌ (`fc_compatible: false`) | Switch motor command format to PWM-like or use specified attitude setpoint output |
| Latency p95 ≤ 20 ms | ✅ baseline ≈ 1 ms | Already there. Don't regress. |
| Marker-loss recovery | ❌ | Falls out of KF predict step (#3) |
