# Control stage playbook

**Owner suggestion:** the other CS dev or whoever's strongest at controls. ~4-6 hours of work.

**Goal:** add the two named unlocks (active yaw + soft-landing) + velocity feedforward from KF. Don't refactor topology — the cascade is already correct.

---

## Baseline (what's there)

Three independent SISO PIDs (x, y, z) → roll/pitch/thrust setpoints → `DefaultAttitudeController` → per-motor throttles. `yaw_rate_cmd = 0`. `LAND` phase pins `thrust_cmd = -1` (bang-bang, forfeits soft-landing bonus).

Cascade is correct: outer position → inner attitude → rates → mixer. **Keep this.**

---

## Ranked work, in order

### 1. Velocity feedforward from KF — 1-2 h, biggest improvement
Add `boat_estimate.velocity` as a feedforward term:
```python
target_v = boat_estimate.get("velocity", np.zeros(3))

# In world frame, before PID:
err_world = target - p_drone
pitch_pid = PID_x(err_body[0], dt, v_meas=v_body[0])
roll_pid  = -PID_y(err_body[1], dt, v_meas=v_body[1])

# Add feedforward as a velocity bias:
v_des_world = err_world * 0.5 + target_v   # 0.5 = position-to-velocity gain
# rotate to body and add to pitch/roll command
v_des_body = R.T @ v_des_world
ff_gain = 0.15   # similar magnitude to PID Kp
pitch_cmd =  pitch_pid + ff_gain * v_des_body[0]
roll_cmd  =  roll_pid  - ff_gain * v_des_body[1]
```
**Rule of thumb:** full FF gain in APPROACH/DESCEND, drop to 0.7 in LAND (estimates noisy near surface). Don't gate FF on confidence below 0.5 — better to track a stale estimate than aim at origin.

**Expected impact:** 30–50% lateral RMS reduction at MEDIUM/HARD boat speeds.

### 2. Soft-landing descent profile — 1 h, directly scores points
Replace `thrust_cmd = -1` in LAND with closed-loop vz tracking:
```python
h = z_above   # drone altitude above marker
v_max = 0.8
h_tau = 0.5
vz_target = -v_max * (1.0 - np.exp(-h / h_tau))   # smooth exponential descent
if h < 0.15:
    vz_target = -0.2   # constant slow touchdown
# Feed vz_target as the z-PID setpoint (replace target_z mechanism)
err_vz = vz_target - v_world[2]
thrust_cmd_soft = clip(Kp_vz * err_vz, -1.0, 1.0)
# Add a thrust floor so we don't free-fall:
thrust_cmd = max(thrust_cmd_soft, -0.7)   # at most 30% below hover
```
Target touchdown velocity ~0.2-0.3 m/s.

**Gate this on `phase_out.get("commit_soft", False)`** so decide() can fall back to bang-bang when conditions aren't met.

### 3. Active yaw control — 1-2 h, unlocks landing condition
Read `boat_estimate.heading` (when not None), compute yaw error, drive `yaw_rate_cmd`:
```python
boat_yaw = boat_estimate.get("heading")
drone_yaw = drone_state["attitude"][2]
if boat_yaw is not None:
    yaw_err = wrap_pi(boat_yaw - drone_yaw)
    # Resolve mod-π ambiguity: pick the heading closer to current drone yaw at first lock,
    # then lock with 20° hysteresis (don't flip).
    if self._yaw_lock_target is None and yaw_locked_from_decide:
        self._yaw_lock_target = boat_yaw   # commit
    Kp_yaw = 1.0
    deadband = np.deg2rad(5.0)
    if abs(yaw_err) < deadband:
        yaw_rate_cmd = 0.0
    else:
        yaw_rate_cmd = clip(Kp_yaw * yaw_err, -0.8, 0.8)   # rad/s
else:
    yaw_rate_cmd = 0.0
```

**Watch out:**
- Weak yaw authority on 10 kg drone → start yawing in APPROACH/YAW_ALIGN, not at touchdown.
- Gyroscopic precession from large rotors → reduce Kp_yaw to ~0.5 during DESCEND/LAND.
- Mod-π ambiguity → use the same heading branch once committed.

### 4. Tilt compensation — 15 min, cheap insurance
```python
roll, pitch, _ = drone_state["attitude"]
tilt_factor = max(np.cos(roll) * np.cos(pitch), 0.5)   # floor to avoid singularity
thrust_cmd /= tilt_factor
```
Prevents altitude sag during aggressive lateral tracking.

### 5. Anti-windup — 30 min
Modify PID:
```python
out_unclipped = Kp*err + Ki*self._i + Kd*d
out = clip(out_unclipped, ±1)
# only integrate if we're not saturated against the error direction
if not (abs(out_unclipped) > 1 and np.sign(err) == np.sign(out_unclipped)):
    self._i = clip(self._i + err*dt, ±self.i_clip)
```
Bump Ki_z to 0.05, Ki_xy to 0.02. Steady wind on MEDIUM/HARD needs this.

---

## Suggested gain bumps

| Axis | Kp (was) | Kp (try) | Ki | Kd |
|---|---|---|---|---|
| x  | 0.15 | **0.25** | 0.02 | 0.50 |
| y  | 0.15 | **0.25** | 0.02 | 0.50 |
| z  | 0.10 | **0.20** | 0.05 | 0.35 |
| yaw | 1.0 | 1.0 (deadband 5°, rate-limit 0.8 rad/s) | 0 | 0 |

Gains are intentionally conservative in the baseline so teams have headroom. The above is a reasonable first push.

---

## HW-readiness "FC-compatible output" — biggest HW bonus
Recommended output format: **attitude-rate setpoint** `(roll_rate, pitch_rate, yaw_rate, thrust_norm)`. This is the industry-standard companion-computer interface (PX4 `OffboardControlMode.body_rate`, Betaflight MSP_SET_RAW_RC equivalent).

- Per-motor PWM: too low-level (mixer is FC's job). Skip.
- Attitude-angle setpoint: hides the rate loop. Less standard for companion-computer setups.
- **Body-rate setpoint: standard, FC-compatible.** This is the one to target for the bonus.

Plan: emit `(roll_rate_des, pitch_rate_des, yaw_rate_des, thrust_norm)` from `control()` and convert to per-motor throttle via the existing `DefaultAttitudeController` — just expose the rate-setpoint output as a side artifact for the FC-compat metric. Specific format depends on what `evaluate.py` checks for `fc_compatible` — read that scorer code before committing to a format.

---

## What to skip

- **MPC** — not worth 12 h. FF + exponential descent captures ~80% of the gain.
- **Refactoring the cascade** — already correct.
- **L1 adaptive / adaptive gains** — diminishing returns and tuning risk.
- **Trajectory optimization** for full landing — too much engineering vs benefit.

---

## Acceptance checklist

- [ ] Velocity feedforward integrated (reads `boat_estimate.velocity`)
- [ ] Soft descent profile in LAND (gated on `commit_soft`)
- [ ] Active yaw with mod-π lock + hysteresis + deadband
- [ ] Tilt compensation on thrust
- [ ] Anti-windup on all three position PIDs
- [ ] Bumped gains tested on EASY without regression
- [ ] Body-rate setpoint emitted for FC-compatibility (or whatever the scorer checks)
- [ ] Latency: all of the above is O(1) math, easily fits in 20 ms

**Implementation order:** FF → soft descent → yaw → tilt-comp → anti-windup. **Ship at hour 6, tune to hour 10, freeze hour 11.**
