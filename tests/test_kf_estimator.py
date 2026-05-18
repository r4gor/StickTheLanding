"""Acceptance tests for the Kalman filter estimator in
``teams/stick-the-landing/agent.py``.

Two tests from teams/stick-the-landing/notes/estimate.md:
    1. Synthetic constant-velocity boat at 1.0 m/s + noisy measurements ->
       filtered velocity converges to within +/-10% of truth within 2 s.
    2. Simulated 1 s camera blackout -> position estimate stays within
       0.3 m of truth (the recovery-bonus criterion in evaluate.py).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = REPO_ROOT / "teams" / "stick-the-landing" / "agent.py"


def _load_team_agent_module():
    """Load teams/stick-the-landing/agent.py as a module. The team file
    isn't on the import path by default — load it explicitly so this test
    is independent of how evaluation/evaluate.py imports it."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("team_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["team_agent"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def team_agent_module():
    return _load_team_agent_module()


def _world_to_tvec(team_agent_module, marker_world, drone_pos, drone_attitude):
    """Inverse of the camera->world transform inside ``estimate()``.

    Camera (OpenCV) -> world basis:
        +X (image right)   = -body_y
        +Y (image down)    = -body_x
        +Z (into scene)    = -body_z

    Marker_world = camera_world + R_cam_to_world @ tvec, so
    tvec = R_cam_to_world^T @ (marker_world - camera_world).
    """
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
    rng, noise_sigma,
):
    tvec = _world_to_tvec(team_agent_module, marker_world, drone_pos, drone_attitude)
    if noise_sigma > 0.0:
        tvec = tvec + rng.normal(0.0, noise_sigma, size=3)
    # rvec is required by the percept schema but only the heading-KF reads
    # it; in the position-convergence tests drone tilt is 0 so the heading
    # filter is gated off anyway. Use a finite placeholder.
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


def test_velocity_converges_for_constant_velocity_boat(team_agent_module):
    """A boat moving at 1.0 m/s along +x with noisy ArUco measurements:
    after 2 s (100 agent steps), filtered velocity must be within
    +/-10% of the true 1.0 m/s."""
    agent = team_agent_module.make_agent()
    DT = agent.DT
    rng = np.random.default_rng(seed=42)

    drone_pos = np.array([0.0, 0.0, 5.0])
    boat_pos = np.array([0.0, 0.0, 0.0])
    boat_vel = np.array([1.0, 0.0, 0.0])

    last = None
    n_steps = int(round(2.0 / DT))
    for _ in range(n_steps):
        percept = _make_percept(
            team_agent_module, boat_pos, drone_pos,
            np.zeros(3), rng, noise_sigma=0.05,
        )
        last = agent.estimate(percept, _make_drone_state(drone_pos))
        boat_pos = boat_pos + boat_vel * DT

    assert last is not None
    assert last["fresh"], "Filter never converged to 'trusted' state"
    vel = np.asarray(last["velocity"])
    assert vel.shape == (3,)
    assert abs(vel[0] - 1.0) < 0.10, (
        f"vx={vel[0]:.3f} not within +/-10% of 1.0"
    )
    assert abs(vel[1]) < 0.10, f"vy={vel[1]:.3f} not near 0"


def test_estimate_survives_one_second_blackout(team_agent_module):
    """After a 1 s camera blackout (50 steps with no detection) the
    predict-only KF must keep the position estimate within 0.3 m of
    truth — this is the recovery-bonus criterion in evaluate.py.

    Also asserts that ``get_last_estimate()`` keeps returning a valid,
    finite dict throughout the blackout (required by contracts.md so
    the scorer's RMSE log doesn't see ``None`` mid-episode)."""
    agent = team_agent_module.make_agent()
    DT = agent.DT
    rng = np.random.default_rng(seed=7)

    drone_pos = np.array([10.0, 0.0, 5.0])
    boat_pos = np.array([0.0, 0.0, 0.0])
    boat_vel = np.array([0.5, 0.0, 0.0])

    # Pre-roll: 100 detections to converge the velocity estimate. One
    # estimate() per loop iteration — predict + (gated) update happens
    # exactly once per simulated step.
    for _ in range(100):
        percept = _make_percept(
            team_agent_module, boat_pos, drone_pos,
            np.zeros(3), rng, noise_sigma=0.03,
        )
        # Mimic act()'s bookkeeping so get_last_estimate() sees the latest dict.
        agent._last_estimate = agent.estimate(percept, _make_drone_state(drone_pos))
        boat_pos = boat_pos + boat_vel * DT

    # 1 s blackout: 50 steps, no detection.
    last = None
    n_blackout = int(round(1.0 / DT))
    for _ in range(n_blackout):
        last = agent.estimate({"detected": False}, _make_drone_state(drone_pos))
        agent._last_estimate = last
        boat_pos = boat_pos + boat_vel * DT

        glm = agent.get_last_estimate()
        assert glm is not None, "get_last_estimate() went None during blackout"
        assert np.all(np.isfinite(glm["position"]))
        assert np.all(np.isfinite(glm["velocity"]))

    assert last is not None
    pos_est = np.asarray(last["position"])
    err = float(np.linalg.norm(pos_est[:2] - boat_pos[:2]))
    assert err < 0.3, (
        f"position error {err:.3f} m after 1 s blackout exceeds 0.3 m "
        f"(true xy={boat_pos[:2]}, est xy={pos_est[:2]})"
    )


def test_contract_schema_matches_contracts_md(team_agent_module):
    """Sanity check on the dict shape estimate() returns — required +
    optional fields per contracts.md."""
    agent = team_agent_module.make_agent()
    # Cold start: position is the search prior, from_prior=True.
    out = agent.estimate({"detected": False}, _make_drone_state([0.0, 0.0, 5.0]))
    required = {"position", "velocity", "fresh", "stale_steps", "from_prior"}
    optional = {
        "heading", "heading_rate", "position_cov", "velocity_cov", "confidence",
    }
    assert required.issubset(out.keys()), (
        f"missing required keys: {required - set(out.keys())}"
    )
    assert optional.issubset(out.keys()), (
        f"missing optional keys: {optional - set(out.keys())}"
    )
    assert out["from_prior"] is True
    assert out["fresh"] is False
    assert np.asarray(out["position"]).shape == (3,)
    assert np.asarray(out["velocity"]).shape == (3,)


def test_nan_percept_does_not_propagate(team_agent_module):
    """A NaN tvec must NOT poison the filter or the output. Per
    contracts.md NaN-safety: emit None / drop the measurement."""
    agent = team_agent_module.make_agent()
    nan_percept = {
        "detected": True,
        "tvec": np.array([np.nan, np.nan, np.nan]),
        "rvec": np.zeros(3),
        "image_points": np.zeros((4, 2), dtype=np.float32),
    }
    out = agent.estimate(nan_percept, _make_drone_state([0.0, 0.0, 5.0]))
    # Filter never initialised because we never saw a valid measurement.
    # Output must still be a valid dict (cold-start prior, all finite).
    assert out["from_prior"] is True
    pos = np.asarray(out["position"], dtype=np.float64)
    assert np.all(np.isfinite(pos))
