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


# ------------------------------------------------------------------ baseline preservation
def test_act_still_works_after_refactor():
    """The act() entrypoint must remain functional — baseline preservation."""
    agent = stl.BaselineAgent()
    action = agent.act(_synthetic_obs())
    assert action.shape == (agent.spec.num_motors,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= 0.0) and np.all(action <= 1.0)
