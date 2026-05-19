# RL Agent — Training Log & Roadmap

## What Has Been Done

### Architecture
- **Kept** the classical perception pipeline (ArUco + `cv2.solvePnP`) and estimation pipeline (Kalman Filter, owned by teammates). These are consumed as-is via the `contracts.md` schemas.
- **Replaced** the baseline FSM (`decide`) + PID (`control`) with a trained **PPO neural network policy** (2×256 MLP, ~130K parameters, ~500 KB on disk).
- The policy outputs `(thrust_norm, roll, pitch, yaw_rate)` setpoints, routed through the organizer's `DefaultAttitudeController` for motor mixing.

### Files Created

| File | Purpose |
|---|---|
| `rl_env.py` | Gymnasium wrapper around `BoatLandingEnv`: 18-dim state vector, 4-dim action space, dense + terminal reward |
| `rl_agent.py` | Inference agent: loads trained model, exposes `act_setpoint()` (+15 FC pts) and `get_last_estimate()` (+15 estimation pts) |
| `train_rl.py` | Training script: PPO (SB3), curriculum learning, parallel envs, TensorBoard logging |
| `models/` | Directory for trained model weights (`landing_policy.zip`) |

### State Vector (18 dims, all normalized to ~[-1, 1])

| Dims | Source | Description |
|---|---|---|
| 0–2 | `perceive` + `estimate` | Relative position boat→drone (body frame) |
| 3–5 | `estimate` + `obs["state"]` | Relative velocity (body frame) |
| 6–7 | `obs["state"]["attitude"]` | Roll, pitch |
| 8 | `obs["state"]` + `estimate.heading` | Yaw alignment error mod π |
| 9 | `obs["state"]["angular_velocity"]` | Yaw rate |
| 10–11 | `estimate["velocity"]` | Boat velocity XY (world frame) |
| 12 | `estimate.get("confidence")` | Estimation confidence (heuristic fallback) |
| 13 | `estimate["stale_steps"]` | Detection staleness |
| 14 | `obs["battery"]` | Battery remaining |
| 15 | `obs["time"]` | Normalized time remaining |
| 16 | computed | Altitude above boat |
| 17 | tracked internally | Max descent velocity so far this episode |

> **Note:** `heading`, `confidence`, and `boat velocity` are currently defaulted (0.0, 0.5, zeros). They will be populated automatically when teammates ship the Kalman Filter upgrade — no code changes needed in the RL agent.

### Reward Design

**Dense (per step):**
- `-0.10 × horiz_distance` — drives approach
- `-0.05 × altitude_error` — drives descent when close
- `-0.15 × yaw_error` — heavy weight, yaw alignment is gating
- `-0.30 × (vz - 0.8)²` when `z < 2m` — soft landing pressure
- `-0.01 × Σ(Δaction)²` — control smoothness
- `-0.002` constant — time cost
- `+0.05` proximity bonus when horizontal < 0.5m, altitude < 1.0m

**Terminal (episode end):**
- `LANDED`: `50 × precision_factor × time_factor × battery_factor` + `5` if max descent velocity < 1.0 m/s
- `CRASHED`: `-20`
- `TIMEOUT` / `OUT_OF_BATTERY`: `-5`

This is the **exact scorer formula** from `AGENT_SCORING.md`, so the policy optimizes directly for the leaderboard metric.

### Key Fixes Made During Session
1. **Seed Incrementing**: Each episode resets with a new seed `(seed + 1) % 2³²`, ensuring every episode has different wind, wave, and camera noise sequences. Without this, every episode was identical (same seed=42 repeated), causing the agent to memorize rather than generalize.
2. **Domain Randomization**: Drone starting position is perturbed by ±1.5m XY and ±0.5m Z at every reset, preventing overfitting to a single spawn point.

### Verified Working
- `agent_mode: setpoint` → **+15 FC-compat bonus** confirmed ✅
- `latency_p95_ms: ~1.2 ms` → **+15 latency bonus** confirmed (budget: 20ms) ✅
- `estimation_rmse_m: 1.55` → estimation pipeline alive ✅
- Training loop: 8K-step sanity test completed (model saved/loaded correctly) ✅
- **Currently training**: 200K steps, EASY scenario, 4 parallel envs, ~90 fps

---

## Next Training Runs

### Run 1 (In Progress) — EASY Baseline
```bash
python teams/stick-the-landing/train_rl.py \
    --scenario easy --timesteps 200000 --n-envs 4
```
- **Goal:** Verify the agent learns to land on a static boat in calm conditions
- **Expected time:** ~30 minutes locally
- **Success criterion:** `ep_rew_mean` rises from -239 → positive; at least one LANDED outcome

### Run 2 — EASY Extended (if Run 1 shows improvement)
```bash
python teams/stick-the-landing/train_rl.py \
    --scenario easy --timesteps 300000 --n-envs 4 --resume
```
- Resume from the Run 1 checkpoint for fine-tuning
- **Goal:** Consistent landings (>50% LANDED rate on EASY)

### Run 3 — MEDIUM Fine-tuning
```bash
python teams/stick-the-landing/train_rl.py \
    --scenario medium --timesteps 300000 --n-envs 4 --resume
```
- Fine-tune the EASY-trained model on the moving boat + wind + waves + camera noise
- **Goal:** Policy adapts to linear boat tracking and yaw alignment under wind
- **Expected time:** ~45 minutes

### Run 4 — HARD Fine-tuning
```bash
python teams/stick-the-landing/train_rl.py \
    --scenario hard --timesteps 500000 --n-envs 4 --resume
```
- Fine-tune on curved trajectory + fog + occlusion + aggressive wind
- **Goal:** Robust performance despite partial observability
- **Expected time:** ~75 minutes

### Run 5 — Full Curriculum (Recommended for Cluster / Final Submission)
```bash
python teams/stick-the-landing/train_rl.py \
    --curriculum --timesteps 500000 --n-envs 4
```
Or on the cluster with more cores:
```bash
python teams/stick-the-landing/train_rl.py \
    --curriculum --timesteps 500000 --n-envs 32
```
- Runs all three stages from scratch in sequence
- On 32 cores this should complete in **under 20 minutes total**
- `--n-envs` should match the number of physical CPU cores on the target machine

### Run 6 — Docker + Reference Sim (Pre-Submission Validation)
```bash
docker pull ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon
docker tag ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon challenge-eval:full

CHALLENGE_IMAGE=challenge-eval:full ./docker/run-local.sh \
    python teams/stick-the-landing/train_rl.py \
    --curriculum --timesteps 500000 --n-envs 4 --use-reference-sim
```
- Trains **inside** the official evaluation Docker container against the reference simulator
- Eliminates any sim-to-sim gap between training and evaluation
- Run this for the final submission model

---

## Evaluation Commands

```bash
# Quick test — single scenario
python evaluation/evaluate.py \
    --agent teams/stick-the-landing/rl_agent.py \
    --scenario easy --headless --seed 42

# Full eval — all scenarios
bash teams/stick-the-landing/eval_all.sh

# Monitor training
tensorboard --logdir teams/stick-the-landing/tb_logs/
# Open http://localhost:6006
```

---

## Teammate Integration Notes

When teammates land the upgraded Kalman Filter / perception improvements:
- **No code changes needed** in `rl_agent.py` or `rl_env.py`
- The state builder already reads from `boat_estimate["velocity"]`, `boat_estimate["heading"]`, `boat_estimate["confidence"]`
- These currently default to zeros/None — once populated, the model **must be retrained** to exploit the richer inputs (run the curriculum again)
- Coordinate with teammates to merge perception branch before the final training run
