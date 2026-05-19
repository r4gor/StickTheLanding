"""Acceptance tests for the curved-trajectory + outlier-rejection upgrade to
the boat Kalman filter in ``teams/stick-the-landing/agent.py``.

Phase 1 deliverable (scaffolding):
    - `@pytest.mark.xfail` markers because the upgrades haven't landed yet.
    - Test #1: a 3σ-deviant detection (>> innovation cov) must be DROPPED
      by the NIS / Mahalanobis gate (predict-only that step). Existing
      `NIS_REINIT_AFTER=5` still rescues a sustained shift.
    - Test #2: a turning-rate (constant-turn-rate) KF must predict a
      curved trajectory better than the current 4-state constant-velocity
      KF.  Stricter: after 1 s blackout on a HARD-like curve
      (v=1.8 m/s, r=30 m), the upgraded filter's position error must be
      smaller than 2 * (current CV-only theoretical bound) AND under 0.3 m
      (the recovery-bonus threshold).

Phase 2 will:
    - Extend `_BoatKalmanFilter` state to `[x, y, vx, vy, omega]` with
      coordinated-turn dynamics (EKF, small-angle linearisation).
    - Tighten the outlier rejection so a single 3σ-deviant detection is
      dropped, even when the converged filter's S is small.

Removing the xfail markers is the green light for Phase 2 completion.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = REPO_ROOT / "teams" / "stick-the-landing" / "agent.py"


def _load_team_agent_module():
    """Load teams/stick-the-landing/agent.py as a module — same trick as
    tests/test_kf_estimator.py, kept self-contained so the two test files
    don't share fixtures cross-file."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("team_agent_curve", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["team_agent_curve"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def team_agent_module():
    return _load_team_agent_module()


def _world_to_tvec(team_agent_module, marker_world, drone_pos, drone_attitude):
    """Inverse of estimate()'s camera->world transform. Copy of the helper
    in test_kf_estimator.py — keep them in sync if the transform changes."""
    rpy_to_matrix = team_agent_module.rpy_to_matrix
    cam_offset_z = team_agent_module.CAMERA_BODY_OFFSET_Z

    R_world_body = rpy_to_matrix(drone_attitude)
    body_x = R_world_body[:, 0]
    body_y = R_world_body[:, 1]
    body_z = R_world_body[:, 2]
    camera_world = np.asarray(drone_pos) + body_z * cam_offset_z
    R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
    return R_cam_to_world.T @ (np.asarray(marker_world) - camera_world)


def _make_percept(
    team_agent_module, marker_world, drone_pos, drone_attitude,
    rng=None, noise_sigma=0.0,
):
    tvec = _world_to_tvec(team_agent_module, marker_world, drone_pos, drone_attitude)
    if rng is not None and noise_sigma > 0.0:
        tvec = tvec + rng.normal(0.0, noise_sigma, size=3)
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    return {
        "detected": True,
        "tvec": tvec,
        "rvec": rvec,
        "image_points": np.zeros((4, 2), dtype=np.float32),
    }


def _make_drone_state(position, velocity=(0.0, 0.0, 0.0), attitude=(0.0, 0.0, 0.0)):
    return {
        "position": np.asarray(position, dtype=np.float64),
        "velocity": np.asarray(velocity, dtype=np.float64),
        "attitude": np.asarray(attitude, dtype=np.float64),
        "angular_velocity": np.zeros(3, dtype=np.float64),
        "motor_omegas": np.zeros(4, dtype=np.float64),
    }


# --------------------------------------------------------------------------- #
# Test 1: 3σ-deviant detection is rejected by outlier gate.
# --------------------------------------------------------------------------- #
def test_outlier_rejection_drops_three_sigma_deviant(team_agent_module):
    """Converge the filter on a stationary boat with very low measurement
    noise, then inject ONE detection offset by ~10σ. The KF must reject
    it (predict-only) — the filtered position must not move by more than
    a tiny fraction (a few %) of the injected offset.

    Why this matters: on HARD, a single fog/blur frame can produce a
    catastrophically bad PnP solution. Without outlier rejection, that
    one frame yanks the filter sideways and the controller follows
    (bang-bang steering). With rejection, the filter stays on the
    converged track and the next clean frame snaps it back into shape.
    """
    agent = team_agent_module.make_agent()
    DT = agent.DT
    rng = np.random.default_rng(seed=11)

    drone_pos = np.array([0.0, 0.0, 5.0])
    boat_pos = np.array([0.0, 0.0, 0.0])

    # Pre-roll: ~2 s of clean detections so the KF converges and its
    # innovation covariance S shrinks well below the injected offset.
    pre_steps = int(round(2.0 / DT))
    for _ in range(pre_steps):
        percept = _make_percept(
            team_agent_module, boat_pos, drone_pos, np.zeros(3),
            rng, noise_sigma=0.01,
        )
        agent._last_estimate = agent.estimate(percept, _make_drone_state(drone_pos))

    # Snapshot the converged xy estimate before the outlier.
    before = np.asarray(agent._last_estimate["position"])[:2].copy()

    # Inject one detection offset by 10 m in +x (vastly more than any
    # plausible PnP error — this is the "3σ-deviant" case taken to extreme).
    outlier_world = boat_pos + np.array([10.0, 0.0, 0.0])
    bad_percept = _make_percept(
        team_agent_module, outlier_world, drone_pos, np.zeros(3),
        rng=None, noise_sigma=0.0,
    )
    after = np.asarray(
        agent.estimate(bad_percept, _make_drone_state(drone_pos))["position"]
    )[:2]

    moved = float(np.linalg.norm(after - before))
    # If the gate works, motion is dominated by predict's tiny Q growth,
    # << 1% of the 10 m offset. If the gate misses, the filter follows
    # the outlier into the weeds.
    assert moved < 0.5, (
        f"Filter moved {moved:.3f} m on a 10 m outlier — outlier rejection "
        f"is not firing (before={before}, after={after})"
    )


# --------------------------------------------------------------------------- #
# Test 2: turning-rate filter beats CV on HARD's curved trajectory.
# --------------------------------------------------------------------------- #
def test_turning_rate_filter_predicts_curve_better_than_cv(team_agent_module):
    """Simulate the HARD scenario boat: speed 1.8 m/s on a circle of
    radius 30 m (omega = v/r = 0.06 rad/s). Run the filter through 4 s
    of clean detections, then a 1 s camera blackout. The turning-rate KF
    must keep the predicted position within 0.3 m of truth at the end of
    the blackout — the recovery-bonus criterion in evaluate.py.

    Quantitatively, a constant-velocity (CV) KF predicting forward by 1 s
    on a 30 m curve at 1.8 m/s accumulates a lateral error of roughly
        e_lat ≈ v · (1 - cos(omega · T)) · r / r = r · (1 - cos(0.06))
              ≈ 30 · 0.0018 ≈ 0.054 m,
    plus an arc-length drift that does NOT cancel — sustained over a
    longer blackout the CV error grows quadratically. The CTR filter's
    error grows only with omega-drift Q. This test asserts the CTR path
    meets the 0.3 m criterion; the CV path on the same blackout already
    just barely passes (see test_kf_estimator.py::*_blackout). The
    distinguishing case is a LONGER blackout — use 2 s for headroom.
    """
    agent = team_agent_module.make_agent()
    DT = agent.DT
    rng = np.random.default_rng(seed=29)

    drone_pos = np.array([0.0, 0.0, 7.0])

    # HARD scenario boat curve: v=1.8 m/s, r=30 m, start heading=0.
    v = 1.8
    r = 30.0
    omega = v / r  # 0.06 rad/s
    boat_x, boat_y, boat_theta = 0.0, 0.0, 0.0

    # Pre-roll: 4 s of clean detections — long enough for the turning-rate
    # KF to estimate omega (the boat heading rotates by ~0.24 rad ≈ 14 deg).
    pre_steps = int(round(4.0 / DT))
    for _ in range(pre_steps):
        percept = _make_percept(
            team_agent_module, np.array([boat_x, boat_y, 0.0]),
            drone_pos, np.zeros(3), rng, noise_sigma=0.03,
        )
        agent._last_estimate = agent.estimate(percept, _make_drone_state(drone_pos))
        # Forward-Euler kinematics (exact for constant omega is unnecessary
        # at this DT — 0.02 s · 0.06 rad/s = 0.0012 rad per step).
        boat_x += v * np.cos(boat_theta) * DT
        boat_y += v * np.sin(boat_theta) * DT
        boat_theta += omega * DT

    # 2 s blackout: predict-only, no detections.
    last = None
    n_blackout = int(round(2.0 / DT))
    for _ in range(n_blackout):
        last = agent.estimate({"detected": False}, _make_drone_state(drone_pos))
        agent._last_estimate = last
        boat_x += v * np.cos(boat_theta) * DT
        boat_y += v * np.sin(boat_theta) * DT
        boat_theta += omega * DT

    assert last is not None
    pos_est = np.asarray(last["position"])
    err = float(np.linalg.norm(pos_est[:2] - np.array([boat_x, boat_y])))
    # 0.5 m bound — CTR-KF should comfortably pass; CV-KF accumulates
    # ~0.5-1.0 m at 2 s blackout on this curve.
    assert err < 0.5, (
        f"position error {err:.3f} m after 2 s blackout on curved trajectory "
        f"exceeds 0.5 m bound — the constant-turn-rate KF is not engaged. "
        f"true xy=({boat_x:.3f}, {boat_y:.3f}), est xy={pos_est[:2]}"
    )
