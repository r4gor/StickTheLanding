"""Control-stage tests for StickTheLanding agent.

Pins the contracts.md guarantees on the control surface:
- `act_setpoint(obs)` returns a finite 4-tuple of (thrust, roll, pitch, yaw_rate)
- `_control_setpoints()` returns a finite dict with the four keys
- `control()` accepts both legacy-string and dict-shaped `phase_out`
- Outputs are guaranteed finite even with degenerate inputs (NaN guard)

Run before every push per teammate brief.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load the team's agent module dynamically (path is teams/stick-the-landing/agent.py).
import importlib.util


def _load_team_agent():
    path = REPO_ROOT / "teams" / "stick-the-landing" / "agent.py"
    spec = importlib.util.spec_from_file_location("stl_agent", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stl = _load_team_agent()


def _synthetic_obs():
    return {
        "camera": np.full((480, 640, 3), 60, dtype=np.uint8),
        "state": {
            "position": np.array([5.0, 0.0, 5.0]),
            "velocity": np.zeros(3),
            "attitude": np.zeros(3),
            "angular_velocity": np.zeros(3),
        },
        "battery": 1.0,
        "time": 0.0,
    }


# ------------------------------------------------------------------ FC-compat: act_setpoint
def test_act_setpoint_method_exists():
    """+15 FC bonus depends on this method existing (evaluate.py:155)."""
    agent = stl.BaselineAgent()
    assert hasattr(agent, "act_setpoint")
    assert callable(agent.act_setpoint)


def test_act_setpoint_returns_finite_4tuple():
    agent = stl.BaselineAgent()
    out = agent.act_setpoint(_synthetic_obs())
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"
    assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
    thrust, roll, pitch, yaw_rate = out
    for name, v in (("thrust", thrust), ("roll", roll),
                    ("pitch", pitch), ("yaw_rate", yaw_rate)):
        assert isinstance(v, float), f"{name} should be float, got {type(v).__name__}"
        assert np.isfinite(v), f"{name} must be finite, got {v}"


def test_act_setpoint_thrust_in_valid_range():
    """Thrust setpoint is normalised in [-1, 1]."""
    agent = stl.BaselineAgent()
    thrust, _, _, _ = agent.act_setpoint(_synthetic_obs())
    assert -1.0 <= thrust <= 1.0


# ------------------------------------------------------------------ _control_setpoints
def test_control_setpoints_returns_dict_with_required_keys():
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    perception = agent.perceive(obs["camera"])
    boat_estimate = agent.estimate(perception, obs["state"])
    sp = agent._control_setpoints(obs["state"], boat_estimate, stl.PHASE_APPROACH)
    for k in ("thrust", "roll", "pitch", "yaw_rate"):
        assert k in sp, f"missing key {k}"
        assert isinstance(sp[k], float)
        assert np.isfinite(sp[k])


def test_control_setpoints_accepts_dict_phase_out():
    """Per contracts.md, phase_out may be a dict in the future. Accept both."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    perception = agent.perceive(obs["camera"])
    boat_estimate = agent.estimate(perception, obs["state"])
    sp_dict = agent._control_setpoints(
        obs["state"], boat_estimate,
        {"phase": stl.PHASE_APPROACH, "commit_soft": False},
    )
    sp_str = agent._control_setpoints(
        obs["state"], boat_estimate, stl.PHASE_APPROACH,
    )
    assert sp_dict.keys() == sp_str.keys()


def test_control_accepts_dict_phase_out():
    """control() should not crash when given a dict-shaped phase_out."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    perception = agent.perceive(obs["camera"])
    boat_estimate = agent.estimate(perception, obs["state"])
    action = agent.control(
        obs["state"], boat_estimate,
        {"phase": stl.PHASE_APPROACH},
    )
    assert action.shape == (agent.spec.num_motors,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= 0.0) and np.all(action <= 1.0)


# ------------------------------------------------------------------ NaN / shape safety
def test_phase_from_handles_unknown_shapes():
    """Defensive helper must never raise on malformed input."""
    assert stl.BaselineAgent._phase_from(stl.PHASE_APPROACH) == stl.PHASE_APPROACH
    assert stl.BaselineAgent._phase_from({"phase": stl.PHASE_LAND}) == stl.PHASE_LAND
    # Unknown shapes fall back to SEARCH rather than raising.
    assert stl.BaselineAgent._phase_from(None) == stl.PHASE_SEARCH
    assert stl.BaselineAgent._phase_from(42) == stl.PHASE_SEARCH
    assert stl.BaselineAgent._phase_from({"not_phase": "X"}) == stl.PHASE_SEARCH


def test_act_setpoint_survives_blank_frame():
    """No marker visible — still returns a finite setpoint."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    for _ in range(50):
        out = agent.act_setpoint(obs)
        for v in out:
            assert np.isfinite(v)


# ------------------------------------------------------------------ soft-landing (item #2)
def test_soft_land_profile_below_descent_threshold():
    """Soft-landing vz profile must stay strictly below the 1.0 m/s scorer cap.

    The +5 soft-landing bonus only triggers when max_descent_velocity < 1.0;
    SOFT_LAND_V_MAX = 0.6 with strict margin even at infinite altitude.
    """
    agent = stl.BaselineAgent()
    for h in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 5.0):
        v_max = agent.SOFT_LAND_V_MAX
        tau = agent.SOFT_LAND_H_TAU
        if h < agent.SOFT_LAND_CRAWL_H:
            vz_target = agent.SOFT_LAND_CRAWL_VZ
        else:
            vz_target = -v_max * (1.0 - np.exp(-h / tau))
        assert abs(vz_target) < 1.0, f"vz_target {vz_target} >= 1.0 at h={h}"


def test_soft_land_thrust_floored():
    """Thrust must never drop below SOFT_LAND_THRUST_FLOOR (no free-fall)."""
    agent = stl.BaselineAgent()
    # Massive positive vz error (drone rocketing up) would push thrust to -1
    # without the floor; verify it clamps.
    t = agent._soft_land_thrust(z_above=0.5, vz_world=10.0)
    assert t >= agent.SOFT_LAND_THRUST_FLOOR
    assert np.isfinite(t)


def test_soft_land_thrust_finite_at_zero_altitude():
    """Edge case: h=0 means vz_target = crawl; thrust must be finite."""
    agent = stl.BaselineAgent()
    t = agent._soft_land_thrust(z_above=0.0, vz_world=-0.1)
    assert np.isfinite(t)


def test_soft_land_thrust_respects_target_descent_rate():
    """phase_out.target_descent_rate should clamp v_max smaller."""
    agent = stl.BaselineAgent()
    # tdr=0.3 caps v_max to 0.3. At z=5m, vz_target ≈ -0.3. If drone is
    # already at -0.3, P loop produces ~0 thrust correction (well above floor).
    t = agent._soft_land_thrust(
        z_above=5.0, vz_world=-0.3, target_descent_rate=0.3,
    )
    assert t > agent.SOFT_LAND_THRUST_FLOOR
    assert np.isfinite(t)


def test_soft_land_thrust_handles_nan_target_descent_rate():
    """NaN/garbage target_descent_rate falls back to SOFT_LAND_V_MAX gracefully."""
    agent = stl.BaselineAgent()
    t = agent._soft_land_thrust(
        z_above=0.5, vz_world=0.0, target_descent_rate=float("nan"),
    )
    assert np.isfinite(t)
    t2 = agent._soft_land_thrust(
        z_above=0.5, vz_world=0.0, target_descent_rate="not a number",
    )
    assert np.isfinite(t2)


def test_land_phase_uses_bang_bang_without_commit_soft():
    """Without phase_out.commit_soft, LAND falls back to legacy bang-bang -1.0."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    perception = agent.perceive(obs["camera"])
    boat_estimate = agent.estimate(perception, obs["state"])
    # String phase_out (baseline) — no commit_soft possible.
    sp = agent._control_setpoints(obs["state"], boat_estimate, stl.PHASE_LAND)
    assert sp["thrust"] == -1.0
    # Dict phase_out with commit_soft=False (or absent) — same fallback.
    sp2 = agent._control_setpoints(
        obs["state"], boat_estimate, {"phase": stl.PHASE_LAND},
    )
    assert sp2["thrust"] == -1.0


def test_land_phase_uses_soft_profile_with_commit_soft():
    """With commit_soft=True, LAND uses soft profile — thrust >= floor and != -1.0."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    # Put drone close to ground, hovering — soft profile should engage.
    obs["state"]["position"] = np.array([0.0, 0.0, 0.5])
    obs["state"]["velocity"] = np.zeros(3)
    perception = agent.perceive(obs["camera"])
    boat_estimate = agent.estimate(perception, obs["state"])
    sp = agent._control_setpoints(
        obs["state"], boat_estimate,
        {"phase": stl.PHASE_LAND, "commit_soft": True},
    )
    assert sp["thrust"] >= agent.SOFT_LAND_THRUST_FLOOR
    assert sp["thrust"] != -1.0  # not legacy bang-bang
    assert np.isfinite(sp["thrust"])


# ------------------------------------------------------------------ velocity feedforward (item #3)
def test_velocity_ff_uses_kf_estimate():
    """FF should bias pitch in the direction of estimated boat motion.

    With drone XY at the same point as the boat position estimate, the only
    forcing on pitch comes from FF — a +x boat velocity must increase pitch
    relative to the zero-velocity baseline.
    """
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    base_estimate = {
        "position": np.array([5.0, 0.0, 0.0]),  # same XY as drone
        "velocity": np.zeros(3),
        "fresh": True,
        "stale_steps": 0,
        "from_prior": False,
    }
    moving_estimate = dict(base_estimate)
    moving_estimate["velocity"] = np.array([2.0, 0.0, 0.0])  # boat moving +x

    sp_base = agent._control_setpoints(obs["state"], base_estimate, stl.PHASE_APPROACH)
    # Reset PIDs so the comparison isn't biased by integrator state.
    agent.pid_x.reset(); agent.pid_y.reset(); agent.pid_z.reset()
    agent._prev_phase = None
    sp_moving = agent._control_setpoints(obs["state"], moving_estimate, stl.PHASE_APPROACH)

    assert sp_moving["pitch"] > sp_base["pitch"], (
        f"pitch did not increase with +x boat velocity: "
        f"base={sp_base['pitch']:.4f}, moving={sp_moving['pitch']:.4f}"
    )


def test_velocity_ff_handles_missing_velocity():
    """Missing 'velocity' key in boat_estimate must default to zero, not crash."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    estimate = {
        "position": np.array([5.0, 0.0, 0.0]),
        "fresh": True,
        "stale_steps": 0,
        "from_prior": False,
    }
    sp = agent._control_setpoints(obs["state"], estimate, stl.PHASE_APPROACH)
    for v in sp.values():
        assert np.isfinite(v)


def test_velocity_ff_handles_nan_velocity():
    """NaN/Inf in boat_estimate.velocity must fall back to zero defaults."""
    agent = stl.BaselineAgent()
    obs = _synthetic_obs()
    estimate = {
        "position": np.array([5.0, 0.0, 0.0]),
        "velocity": np.array([float("nan"), float("inf"), 0.0]),
        "fresh": True,
        "stale_steps": 0,
        "from_prior": False,
    }
    sp = agent._control_setpoints(obs["state"], estimate, stl.PHASE_APPROACH)
    for v in sp.values():
        assert np.isfinite(v)


def test_velocity_ff_gain_lower_in_land():
    """LAND phase uses 0.7x FF gain — gentler near touchdown."""
    agent = stl.BaselineAgent()
    assert agent.VEL_FF_GAIN_LAND < agent.VEL_FF_GAIN_APPROACH
    assert abs(agent.VEL_FF_GAIN_LAND - 0.7 * agent.VEL_FF_GAIN_APPROACH) < 1e-9


# ------------------------------------------------------------------ baseline preservation
def test_act_still_works_after_refactor():
    """The act() entrypoint must remain functional — baseline preservation."""
    agent = stl.BaselineAgent()
    action = agent.act(_synthetic_obs())
    assert action.shape == (agent.spec.num_motors,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= 0.0) and np.all(action <= 1.0)
