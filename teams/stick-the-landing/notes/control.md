# Control stage playbook

**Owner suggestion:** the other CS dev or whoever's strongest at controls. ~4-6 hours of work.

**Goal:** add velocity feedforward, soft-landing profile, active yaw, and `act_setpoint()` for +15 FC-compat. Don't refactor topology — the cascade is already correct.

> **Research validation (2026-05-18):** The current cascade (position-PID → roll/pitch/thrust setpoint → attitude controller → mixer) matches PX4 and ArduPilot exactly ([PX4 controller diagrams](https://docs.px4.io/main/en/flight_stack/controller_diagrams), [AC_PosControl.cpp](https://github.com/ArduPilot/ardupilot/blob/master/libraries/AC_AttitudeControl/AC_PosControl.cpp)). SE(3) geometric control gains marginal benefit for non-aggressive maneuvers — **don't refactor**. MPC: TinyMPC on Crazyflie took a research team months ([Bitcraze post 2024-07](https://www.bitcraze.io/2024/07/bringing-model-predictive-control-to-the-crazyflie-with-tinympc/)); skip is correct for 12h.

---

## Baseline (what's there)

Three independent SISO PIDs (x, y, z) → roll/pitch/thrust setpoints → `DefaultAttitudeController` → per-motor throttles. `yaw_rate_cmd = 0`. `LAND` phase pins `thrust_cmd = -1` (bang-bang, forfeits soft-landing bonus).

Cascade is correct: outer position → inner attitude → rates → mixer. **Keep this.**

---

## What's already in the stock `DefaultAttitudeController` (don't reinvent)

From `boat_landing/controllers.py`:
- **Inner topology**: PD on attitude → P on yaw rate → inverse mixer.
- **Gains are data-driven** from `DroneSpec.inertia`: `Kp_roll = Ixx · ω_n_roll²`, `Kd = 2·ζ·ω_n·I`, with defaults `ω_n_roll=8.0`, `ω_n_pitch=6.0`, `ω_n_yaw_rate=4.0`, `ζ=0.7`. **Tunable** via the constructor.
- **Gyroscopic precession FF is already in there** — uses `state.motor_omegas` if present, otherwise hover-omega estimate. The cross term `ω_body × (rotor_inertia · spin · ω_motor · axis)` is summed into the torque setpoint. So as long as `drone_sim.py` populates `motor_omegas`, gyroscopic effects are pre-handled.
- **VTOL airframe**: 4 motors at body positions `[±1.0, ±0.85, 0]`, spins alternating, `k_T = 4e-4`, `k_Q = 8e-6` → `k_Q/k_T = 0.020` (**low yaw authority**). `Izz = 5.74` (high inertia in yaw). Yaw transients settle in ~2 s — plan yaw alignment early.

---

## Ranked work, in order

### 1. Refactor `control()` into `_control_setpoints()` + expose `act_setpoint()` — 30 min, +15 pts FC-compat
This is the **biggest single-line-count improvement to score.** Per `evaluate.py:146-163`, `fc_compatible` is awarded for **the mere presence** of an `act_setpoint(obs)` method returning `(thrust_norm, roll, pitch, yaw_rate)`. Not a format validation.

```python
def _control_setpoints(self, state, boat_estimate, phase_out):
    # everything currently in control() that computes (thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd)
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

That's it. +15 pts on every LANDED run.

### 2. Velocity feedforward from KF — 1-2 h, biggest tracking improvement
Add `boat_estimate.velocity` as a feedforward term:
```python
target_v = boat_estimate.get("velocity", np.zeros(3))

err_world = target - p_drone
pitch_pid = PID_x(err_body[0], dt, v_meas=v_body[0])
roll_pid  = -PID_y(err_body[1], dt, v_meas=v_body[1])

# Add feedforward as a velocity bias:
v_des_world = err_world * 0.5 + target_v   # 0.5 = position-to-velocity gain
v_des_body  = R.T @ v_des_world
ff_gain = 0.15
pitch_cmd =  pitch_pid + ff_gain * v_des_body[0]
roll_cmd  =  roll_pid  - ff_gain * v_des_body[1]
```
Full FF gain in APPROACH/DESCEND, drop to 0.7× in LAND. Don't gate below confidence 0.5 — tracking a stale estimate beats aiming at origin.

**Expected impact:** 30-50% lateral RMS reduction at MEDIUM/HARD boat speeds.

### 3. Acceleration feedforward (new from research) — 30 min
ArduPilot's `AC_PosControl` uses position-P + velocity-PID + **accel-FF**. Cheap addition once the velocity FF is in:
```python
target_a = (target_v - self._prev_target_v) / dt   # smooth lightly
self._prev_target_v = target_v
# Add to roll/pitch via small accel gain (a → tilt: roll = a_y/g)
accel_gain = 0.05
pitch_cmd += accel_gain * (R.T @ target_a)[0] / 9.81
roll_cmd  -= accel_gain * (R.T @ target_a)[1] / 9.81
```
Source: [AC_PosControl.cpp](https://github.com/ArduPilot/ardupilot/blob/master/libraries/AC_AttitudeControl/AC_PosControl.cpp). Cheap, meaningful win on the curved HARD scenario.

### 4. Soft-landing descent profile — 1 h, directly scores +5 pts
Replace `thrust_cmd = -1` in LAND with closed-loop vz tracking. **Strict `<1.0 m/s`** for the bonus, so target ~0.3 m/s with margin.
```python
h = z_above
v_max = 0.6                                    # was 0.8, lowered per research
h_tau = 0.5
vz_target = -v_max * (1.0 - np.exp(-h / h_tau))
if h < 0.10:
    vz_target = -0.15                          # was -0.2, tightened per research
err_vz = vz_target - v_world[2]
thrust_cmd_soft = clip(Kp_vz * err_vz, -1.0, 1.0)
thrust_cmd = max(thrust_cmd_soft, -0.7)        # floor to avoid free-fall
```

Literature: exponential is fine; minimum-jerk polynomial scores slightly higher in moving-platform challenges ([IFAC 2024 Min-Jerk Soft Landing](https://www.sciencedirect.com/science/article/pii/S2405896324000193)) but isn't worth the dev cost. PX4 uses ~0.3 m/s for its `MPC_LAND_CRWL` final-stage crawl ([PX4 Position Mode](https://docs.px4.io/main/en/flight_modes_mc/position)).

**Gate on `phase_out.get("commit_soft", False)`** — when not soft-committed, fall back to baseline bang-bang.

### 5. Active yaw control — 1-2 h, unlocks landing condition
Read `boat_estimate.heading` and decide-stage's `yaw_target_locked` flag:
```python
boat_yaw = boat_estimate.get("heading")
drone_yaw = drone_state["attitude"][2]

if boat_yaw is not None and phase_out.get("yaw_locked"):
    yaw_target = phase_out.get("yaw_target", boat_yaw)   # decide may freeze the target
    yaw_err = wrap_pi(yaw_target - drone_yaw)
    deadband = np.deg2rad(8.0)                            # was 5°, widened per research
    if abs(yaw_err) < deadband:
        yaw_rate_cmd = 0.0
    else:
        Kp_yaw = 1.0
        # Reduce gain during descent: weak yaw + integrator-driven thrash bleeds roll/pitch
        if phase_out["phase"] in ("DESCEND", "LAND"):
            Kp_yaw = 0.5
        yaw_rate_max = 1.5 if phase_out["phase"] == "APPROACH" else 0.4   # per research
        yaw_rate_cmd = clip(Kp_yaw * yaw_err, -yaw_rate_max, yaw_rate_max)
else:
    yaw_rate_cmd = 0.0
```

**Key research-confirmed gotchas:**
- **VTOL has weak yaw** (`k_Q/k_T = 0.020`, high `Izz`) — start yawing in APPROACH, not at touchdown. Confirmed by [agile UAV ship landing arXiv 2411.05445](https://arxiv.org/pdf/2411.05445).
- **Freeze yaw target once locked** (handled in `decide()`). Don't keep tracking down to touchdown.
- **Mod-π ambiguity from ArUco**: commit on first lock with a hysteresis ([opencv_contrib #3190](https://github.com/opencv/opencv_contrib/issues/3190), [arXiv 1909.11888](https://arxiv.org/pdf/1909.11888)).
- **Gyroscopic precession**: already cancelled by the stock attitude controller's FF when `motor_omegas` is populated. Doesn't need extra work here.

### 6. Tilt compensation — 15 min, cheap insurance
```python
roll, pitch, _ = drone_state["attitude"]
tilt_factor = max(np.cos(roll) * np.cos(pitch), 0.5)   # 60° tilt floor
thrust_cmd /= tilt_factor
```
Prevents altitude sag during aggressive lateral tracking. Standard altitude-control correction ([Stanford Quadrotor Dynamics, Hoffmann et al.](https://ai.stanford.edu/~gabeh/papers/Quadrotor_Dynamics_GNC07.pdf)).

### 7. Anti-windup — 30 min
```python
out_unclipped = Kp*err + Ki*self._i + Kd*d
out = clip(out_unclipped, ±1)
# clamp-condition: don't integrate further into saturation
if not (abs(out_unclipped) > 1 and np.sign(err) == np.sign(out_unclipped)):
    self._i = clip(self._i + err*dt, ±self.i_clip)
```
Bump Ki_z to 0.05, Ki_xy to 0.02. Reset integrators on phase transitions (already done in baseline).

---

## Suggested gain bumps

| Axis | Kp (baseline) | Kp (try) | Ki | Kd |
|---|---|---|---|---|
| x  | 0.15 | **0.25** | 0.02 | 0.50 |
| y  | 0.15 | **0.25** | 0.02 | 0.50 |
| z  | 0.10 | **0.20** | 0.05 | 0.35 |
| yaw | 1.0 (unused in baseline) | 1.0 (APPROACH) → 0.5 (DESCEND/LAND), deadband 8° | **0** | 0 |

Baseline gains are intentionally conservative — there's headroom.

---

## HW-readiness "FC-compatible output" — solved by item 1 above

Implementing `act_setpoint(obs)` returning `(thrust_norm, roll, pitch, yaw_rate)` is **all that's required**. This is closer to PX4's `SET_ATTITUDE_TARGET` (attitude-quaternion + thrust) than to body-rate output, and is the standard companion-computer interface ([PX4 Offboard Mode](https://docs.px4.io/main/en/flight_modes/offboard)). Body-rate output (what draft-1 originally recommended) is more racing-grade and is **not** what the scorer wants — the format check is "method exists" rather than "output is body-rate-shaped".

---

## What to skip

- **MPC** — TinyMPC/Acados RTI require significant tuning + integration time. FF + exponential descent captures ~80% of the gain ([TinyMPC arXiv 2310.16985](https://arxiv.org/pdf/2310.16985)).
- **Refactoring the cascade** — confirmed correct by both PX4 and ArduPilot.
- **L1 adaptive / adaptive gains** — diminishing returns, tuning risk.
- **Trajectory optimization for full landing** — too much engineering vs benefit.
- **Body-rate output instead of attitude-setpoint** — draft-1's recommendation was wrong; presence of `act_setpoint` is what scores.

---

## Acceptance checklist

- [ ] `_control_setpoints()` extracted from `control()`, returns 4-tuple
- [ ] `act_setpoint(obs)` exposed on the Agent class
- [ ] Velocity feedforward integrated (reads `boat_estimate.velocity`)
- [ ] Acceleration feedforward (cheap; do after velocity FF)
- [ ] Soft descent profile in LAND (gated on `commit_soft`)
- [ ] Active yaw with mod-π lock + hysteresis + 8° deadband + gain-scheduled by phase
- [ ] Tilt compensation on thrust
- [ ] Anti-windup on all three position PIDs
- [ ] Bumped gains tested on EASY without regression
- [ ] Every output is finite (NaN guard at the end of `_control_setpoints`)
- [ ] Latency: all O(1) math, easily fits in 20 ms

**Implementation order:** `act_setpoint` refactor → velocity FF → soft descent → yaw → tilt-comp → anti-windup → accel FF (optional). **Ship at hour 6, tune to hour 10, freeze hour 11.**
