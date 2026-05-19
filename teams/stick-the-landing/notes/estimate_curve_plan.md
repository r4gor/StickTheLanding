# estimate-curve plan — turning-rate KF + outlier rejection

**Branch:** `estimate-curve` (off `checkpoint-v2`)
**Owner:** TEAMMATE `estimate-curve`
**Time budget:** 1h45m
**Lands ON TOP of:** `estimate-v2` (dual-source heading) after lead cherry-picks.

## Context

HARD scenario boat is on a constant-radius curve (r=30 m at v=1.8 m/s,
omega = v/r = 0.06 rad/s). The current `_BoatKalmanFilter` uses a
constant-velocity (CV) process model — it absorbs the 0.11 m/s²
centripetal acceleration into Q but is biased during sustained turns
(estimate.md cites a <0.1 m bias on RMSE on HARD; ok for the estimation
bonus but it hurts the *control* loop during turns and inflates blackout
prediction error on the recovery path).

Two upgrades, additive only (estimate.md design contract preserved):

1. **Turning-rate (coordinated-turn) EKF.** State `[x, y, vx, vy, omega]`.
   - Process model continuous: `dx=vx, dy=vy, dvx=-vy·omega, dvy=vx·omega, domega=0`.
   - For dt=0.02 s, omega ≈ 0.06 rad/s → omega·dt ≈ 0.0012 rad — small-angle
     linearisation is plenty accurate. EKF Jacobian F = ∂f/∂x evaluated at
     current state. Measurement model unchanged (H = 2×5 selecting x,y).
   - Initial omega = 0. Q for omega: σ_omegadot ≈ 0.02 rad/s² (boats turn
     slowly).
2. **Outlier rejection.** If `NIS > 9.0` (≈ chi-square 2-DoF 99% / Mahalanobis
   ~3σ-equivalent), DROP the detection (predict-only that step). The existing
   `NIS_REINIT_AFTER=5` already recovers from a sustained shift. Keep the
   13.8 (99.9%) gate as the *reinit* trigger so we still reseed on long
   dropouts; the 9.0 gate is the *drop-only* path that protects converged
   tracks from single-frame catastrophes.

## Phase 1 (T+0 to T+0:30) — RESEARCH + TEST SCAFFOLDING (this commit)

- [x] Read `notes/estimate.md`, `docs/TIPS.md#improvement-1`, `scenarios/hard.yaml`.
- [x] Read `_BoatKalmanFilter` (agent.py lines ~97-163) and the estimate()
      flow (lines ~785-980).
- [x] Write `tests/test_estimate_curve.py` with `@pytest.mark.xfail`:
      `test_outlier_rejection_drops_three_sigma_deviant` and
      `test_turning_rate_filter_predicts_curve_better_than_cv`.
- [x] Write this plan.
- [ ] Commit + push phase 1.
- [ ] SendMessage team-lead.

## Phase 2 (T+0:30 to T+1:45) — IMPLEMENT

1. `git fetch origin && git rebase origin/main` (pick up estimate-v2).
2. Extend `_BoatKalmanFilter`:
   - State dim 4 → 5: append `omega`.
   - `_F(dt, state)` → linearised Jacobian (EKF predict).
   - `_Q(dt)` → 5×5 block: 4×4 CV block as today + scalar omega white-noise
     block on `[4,4]` = `σ_omegadot² · dt`. Cross-terms zero.
   - `H` stays `[[1,0,0,0,0],[0,1,0,0,0]]`.
   - `predict(dt)`: nonlinear state propagation + linear Jacobian P update.
     For dt=0.02 s and |omega·dt|<<1, use closed-form small-angle update
     of vx,vy or even simpler `vx -= vy·omega·dt; vy += vx_old·omega·dt`
     (Heun-style for second-order accuracy if cheap).
   - `initialize(z, sigma_xy)`: seed omega=0, P_omega = (0.1 rad/s)² so
     it's free to roam quickly when the boat starts turning.
3. Add outlier rejection in `estimate()` step 4:
   - New constant `NIS_DROP_THRESHOLD = 9.0` (chi-square 2-DoF 99%, ≈ 3σ
     in Mahalanobis-equivalent terms for 2D).
   - If `9.0 < NIS ≤ 13.8`: predict-only this step, increment a *soft*
     reject counter (separate from the existing hard-reject counter that
     trips the reinit). Don't reseed for soft rejects.
   - If `NIS > 13.8`: existing path (hard reject; reinit after 5).
4. Validate:
   - `pytest tests/test_estimate_curve.py tests/test_kf_estimator.py -v`
     — both must pass.
   - `python evaluation/evaluate.py --agent teams/stick-the-landing/agent.py
      --scenario hard --headless --seed 42 --output /tmp/scores_estimate_curve.json`
     — confirm `est_bonus ≥ 14.0` and HARD landing_score ↑.
   - Spot-check EASY/MEDIUM hadn't regressed (run if budget allows).
5. SendMessage team-lead with results.

## Compliance checklist (re-stated from lead brief)

- [ ] Additive only. `estimate()` signature unchanged.
- [ ] All optional fields still emitted; new state never propagates NaN
      (omega is finite by construction; guard the Jacobian for huge omega).
- [ ] `tests/test_kf_estimator.py` MUST still pass (no regression):
      - cv-converges-1-mps test stays green (omega gets pushed to 0 by Q
        when boat moves in straight line).
      - 1-s-blackout test stays green (CTR-predict on omega≈0 is identical
        to CV-predict).
      - schema + NaN tests stay green.
- [ ] HARD est_bonus ≥ 14.0 baseline, landing_score improved.
- [ ] Don't touch perceive/decide/control or `_HeadingKalmanFilter`.

## Risk register

- **EKF Jacobian sign error.** Catch with the CV-equivalence regression
  (omega=0 should reproduce CV exactly to machine precision).
- **NIS_DROP=9.0 too tight under fog.** If we see more rejects than
  expected on HARD eval, fall back to 11 (97.5%) — both still hit
  `_REINIT_AFTER=5` for sustained shifts. Decide based on the eval log.
- **omega estimate noisy on EASY (static boat).** Q on omega is tiny;
  omega should converge to ~0 and stay there. If the test_kf_estimator
  CV-convergence test fails, dial σ_omegadot down further.
- **Coordination with estimate-v2.** Their work touches the heading
  filter; ours touches only the position filter. Rebase conflicts are
  unlikely but I'll resolve them in their favour for the heading-KF
  block and re-run all tests after the rebase.

## Acceptance gates (Phase 2 done = all green)

1. `pytest tests/test_estimate_curve.py -v` → 2 PASS (xfail markers removed).
2. `pytest tests/test_kf_estimator.py -v` → 4 PASS (no regression).
3. `evaluate.py --scenario hard --seed 42` → est_bonus ≥ 14.0, landing
   score improved over `checkpoint-v2` baseline.
4. Git push to `origin/estimate-curve`, SendMessage team-lead.
