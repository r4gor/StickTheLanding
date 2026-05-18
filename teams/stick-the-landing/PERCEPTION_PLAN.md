# Perception — Development Plan

**Owner:** Person 1 (Christian) · **Deliverable:** the `perceive()` method of the team's
`agent.py`, plus the module-level `marker_to_world()` helper.

## Approach: one `agent.py`, no separate module

The submission is a single `agent.py` (+ `drone_sim.py`, `submission.yaml`). Perception is
**not** a separate file — a helper module would force `agent.py` to patch `sys.path` and
import into the global namespace, bypassing the evaluator's per-agent isolation.

`agent.py` is a fork of `agents/agent_template.py` — the documented starting point, with
`act()` locked and `perceive / estimate / decide / control` as blank stubs. Perception
work = **fill in the `perceive()` method** and add the module-level `marker_to_world()`
helper. `estimate / decide / control` stay as template stubs until Persons 2 & 3
implement them — so the agent does not land until then (the intended template workflow).

## Files

| File | Purpose | Shipped? |
|---|---|---|
| `teams/stick-the-landing/agent.py` | The agent — shared with Persons 2 & 3 | **Yes** |
| `teams/stick-the-landing/check_perception.py` | Ground-truth dev harness (reads `info` — legal in a dev script, never in the agent) | No |
| `teams/stick-the-landing/test_perception.py` | pytest unit tests | No |

## What perception owns

`perceive(camera_image) -> dict` — pure image work, once per 50 Hz step:
- ArUco detection (`DICT_5X5_100`, marker ID 0) + `solvePnP` (`IPPE_SQUARE`).
- CLAHE + subpixel refinement + widened adaptive threshold for fog/blur robustness.
- Stale-frame detection (frame hash), NaN / FOV-edge guards, reprojection-error quality.

`marker_to_world(perception, drone_state) -> (marker_world, marker_yaw)` — module-level
helper holding the camera→world transform. Person 2's `estimate()` calls it.

Perception never smooths or predicts — filtering is Person 2's Kalman filter.

## Interface contract (lock with Person 2)

`perceive()` returns: always `detected`, `stale`; when detected also `tvec`, `rvec`,
`image_points`, `reproj_error`, `marker_px`.

`marker_to_world(perception, drone_state) -> (marker_world (3,) | None, marker_yaw | None)`.

## Milestones

### M0 — Scaffold  ✅
- [x] Copy `agent.py` from `agents/agent_template.py`; fix `REPO_ROOT` to `parents[2]`.
- [x] Create `check_perception.py` + `test_perception.py`.

### M1 — Core detection  ✅
- [x] ArUco + `solvePnP` (IPPE_SQUARE), select marker ID 0, NaN / depth guards.

### M2 — Transform + heading  ✅
- [x] `marker_to_world()`: `R_cam_world = [-by, -bx, -bz]`, camera offset, heading from `rvec`.

### M3 — Integration with Person 2  (sprint H4–H6)
- [ ] Person 2's `estimate()` calls `marker_to_world()`, then runs the Kalman filter.
- [ ] First full-agent run on MEDIUM → target `LANDED`.

### M4 — Robustness  ✅ (verify on HARD)
- [x] CLAHE, subpixel refinement, widened adaptive threshold, stale-frame flag, quality metric.
- [ ] Confirm detection survives HARD fog/blur; recovery blackout → clean no-detections.

### M5 — Tuning  (sprint H8–H10)
- [ ] Seed sweep; fix the worst failure mode; freeze parameters.

## Tests

```bash
# unit tests (synthetic, fast)
./docker/run-local.sh pytest teams/stick-the-landing/test_perception.py -v
# ground-truth accuracy harness (flies the baseline, scores perception per scenario)
./docker/run-local.sh python teams/stick-the-landing/check_perception.py
```

`check_perception.py` reports, per scenario, detection rate and xy RMS error vs ground
truth. **Targets:** detection rate > 0.7 (EASY/MEDIUM, away from the close-descent tail);
xy RMS error < 0.15 m (EASY) / 0.2 m (MEDIUM).

## Known dropout regimes — expected, not bugs

- **Close-range overfill** below ~0.8 m altitude (marker overfills the 90° FOV).
- **FOV-edge** near-collinear corners → IPPE NaN (guarded).
- **Occlusion** blackout (recovery scenario).

Report `detected: False` in all three — Person 2's KF coasts on prediction.

## Constraints

- The agent never reads `info`, scenario YAML, or `boat_landing.*` internals. Only
  `check_perception.py` (dev-only) may read `info`.
- Debug output to **stderr**. Keep `perceive()` allocation-light (latency budget).
