# Sim pipeline — data contract + step() math

Mathy reference for `boat_landing/drone_interface.py` (contract) and `agents/drone_sim_baseline.py` (reference implementation). Use this to know **where each Tier-1 feature hooks in** without re-reading 660+ lines.

---

## 1. `boat_landing/drone_interface.py` — data contract

### Input (one-time): `drones/vtol.yaml` → `DroneSpec`

Loaded via `load_drone_spec(path)`. Read-only from your sim's perspective.

| Symbol | Type | Source field | Used by |
|---|---|---|---|
| `m` | float | `mass` | gravity, accel |
| `I` | (3,3) | `inertia.{Ixx,Iyy,Izz}` | angular dynamics |
| `r_i` | (N,3) | `motors[i].position` | torque arm |
| `a_i` | (N,3) unit | `motors[i].thrust_axis` | thrust direction |
| `s_i` | (N,) ±1 | `motors[i].spin` | drag-torque sign |
| `k_T` | float | `propeller.thrust_coefficient` | `T = k_T·ω²` |
| `k_Q` | float | `propeller.drag_coefficient` | `Q = k_Q·ω²` |
| `ω_max` | float | `motor.omega_max` | throttle → ω_cmd |
| `τ_m` | float | `motor.time_constant` | motor lag |
| `V_nom, R_int, C_Wh` | float | `battery.*` | **unused by baseline → T1.B hook** |
| `c_lin, c_quad` | (3,) each | `aero.drag_{linear,quadratic}` | **unused → T1.C hook** |

### Per-step in (env → sim): `step(u, F_ext, dt, τ_ext=0, z_ground=None)`

| Arg | Shape | Meaning |
|---|---|---|
| `u` | (N,) ∈ [0,1] | per-motor throttle |
| `F_ext` | (3,) world | wind force |
| `dt` | float | step size [s] |
| `τ_ext` | (3,) world or `None` | external torque (rare) |
| `z_ground` | float or `None` | world-z of surface below — **T1.F ground_effect input** |

### Out (sim → env): `get_state() → DroneState`

```
position p ∈ ℝ³ (world)
velocity v ∈ ℝ³ (world)
quaternion q = (x, y, z, w)
ω_body ∈ ℝ³ (body frame)
motor_omegas ω ∈ ℝᴺ ← MUST be populated for T1.A & T1.B scoring
```

---

## 2. `agents/drone_sim_baseline.py` — `step()` pipeline

Sub-step loop: `n_sub = ⌈dt / 5 ms⌉`, `dt_sub = dt / n_sub`. Each sub-step runs stages 1–9:

| # | Stage | Math | Tier-1 hook |
|---|---|---|---|
| 1 | **Motor lag** | `ω_cmd = u · ω_max` → `ω ← ω + (1 − e^(−dt_sub/τ_m))·(ω_cmd − ω)` | T1.A ✓ baseline. **T1.B: scale ω_cmd by V_bat/V_nom** |
| 2 | **Per-motor wrench** | `T_i = k_T · ω_i²`, `Q_i = k_Q · ω_i²` <br> `F_i = T_i · a_i` (body) | **T1.F: T_i ← T_i · (1 + κ·(R/h)²)**, where `h = z − z_ground` |
| 3 | **Mixer** | `F_body = Σ_i F_i` <br> `τ_body = Σ_i (r_i × F_i) + Σ_i (−s_i · Q_i · a_i)` | — |
| 4 | **Rotate body→world** | `R = R(q)` (3×3) | — |
| 5 | **World force** | `F_world = R · F_body + (0, 0, −mg) + F_ext` | **T1.C: F_world += R · (−c_lin ⊙ v_body − c_quad ⊙ \|v_body\| ⊙ v_body)** |
| 6 | **Body torque** | `τ_body_total = τ_body + Rᵀ · τ_ext` | — |
| 7 | **Linear accel** | `a = F_world / m` | — |
| 8 | **Angular accel** (Newton-Euler) | `α_body = I⁻¹ · (τ_body_total − ω_body × (I·ω_body))` | T1.D ✓ baseline (the `ω × Iω` term) |
| 9 | **Integrate (semi-implicit Euler)** | `v ← v + a·dt_sub`, `p ← p + v·dt_sub` <br> `ω_body ← ω_body + α_body·dt_sub` <br> `q ← q ⊗ Δq`, `Δq = (axis·sin(θ/2), cos(θ/2))`, `axis = ω̂_body`, `θ = ‖ω_body‖·dt_sub` | T1.E ✓ baseline (sub-stepping itself) |

### Method-level structure (already split for forks)

```
step()
  └─ _integrate_substep()
       ├─ _motor_dynamics_step()      ← T1.A here; T1.B hooks here
       ├─ _compute_motor_wrench()     ← T1.C, T1.F hook here (or just after)
       ├─ _rotation_matrix(q)
       └─ semi-implicit Euler + _integrate_quaternion()
```

---

## Tier-1 leverage summary

| Feature | Status | Effort | Where |
|---|---|---|---|
| **T1.A motor_lag** | ✓ baseline | — | `_motor_dynamics_step` |
| **T1.D cross_coupling** | ✓ baseline | — | stage 8 |
| **T1.E substepping** | ✓ baseline | — | `step()` loop |
| **T1.B battery_sag** | TODO | scale `ω_cmd` by `V_bat/V_nom`; integrate battery state | `_motor_dynamics_step` |
| **T1.C aero_drag** | TODO | linear + quadratic drag in body frame | stage 5 |
| **T1.F ground_effect** | TODO | per-motor thrust amplification `1 + κ(R/h)²` | stage 2 |

Three TODOs → +15 pts, no restructuring. Read the corresponding `evaluation/sim_validation/t1_*.py` for the **exact** spec each test enforces — implement against the test.

### Anti-bluff gotcha

The sim-scorer compares your `drone_sim.py` byte-identical to `drone_sim_baseline.py`. If they match, `total_score: 0` regardless of declared features. Adding any of the three TODOs above naturally breaks the byte match — but a "trivial whitespace-only diff" wouldn't be enough. The diff has to be a real modification.
