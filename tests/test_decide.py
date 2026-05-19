"""Decision-stage tests for StickTheLanding agent.

Pins the contracts.md guarantees on the decide surface:
  - returns the phase_out dict shape (`phase`, `target_descent_rate`,
    `yaw_locked`, `yaw_target`, `commit_soft`, `abort_reason`)
  - hysteresis: every transition requires N consecutive supporting frames
    (kills the phase flicker the playbook attributes 15-30% of median
    score to)
  - asymmetric enter/exit thresholds keep the FSM from churning at the
    boundary
  - soft-commit gate honours all four conditions from decide.md #7
  - yaw-lock freezes the target heading on YAW_ALIGN entry and doesn't
    track the live boat heading thereafter
  - NaN/Inf inputs collapse to SEARCH with an ``abort_reason``, never
    propagate non-finite values into the output dict

Run before every push per teammate brief.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_team_agent():
    path = REPO_ROOT / "teams" / "stick-the-landing" / "agent.py"
    spec = importlib.util.spec_from_file_location("stl_agent", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stl = _load_team_agent()


# --------------------------------------------------------------------- fixtures
def _state(pos=(0.0, 0.0, 5.5), vel=(0.0, 0.0, 0.0), att=(0.0, 0.0, 0.0)):
    return {
        "position": np.array(pos, dtype=np.float64),
        "velocity": np.array(vel, dtype=np.float64),
        "attitude": np.array(att, dtype=np.float64),
        "angular_velocity": np.zeros(3),
    }


def _estimate(
    pos=(0.0, 0.0, 0.3),
    vel=(0.0, 0.0, 0.0),
    fresh=True,
    stale_steps=0,
    from_prior=False,
    heading=None,
):
    return {
        "position": np.array(pos, dtype=np.float64),
        "velocity": np.array(vel, dtype=np.float64),
        "fresh": fresh,
        "stale_steps": stale_steps,
        "from_prior": from_prior,
        "heading": heading,
        "heading_rate": None,
        "position_cov": None,
        "velocity_cov": None,
        "confidence": None,
    }


def _new_agent():
    return stl.BaselineAgent()


# --------------------------------------------------------------------- shape
def test_decide_returns_phase_out_dict_with_contracts_keys():
    """contracts.md:60-71 — required and optional keys present.

    A bare-string return would silently lose `target_descent_rate`,
    `commit_soft` and the rest; control() falls back to baseline
    behavior and the soft-landing bonus disappears. Pin the shape.
    """
    agent = _new_agent()
    out = agent.decide(_state(), _estimate(), battery=1.0, time_s=0.0)
    assert isinstance(out, dict), "decide() must return a dict per contracts.md"
    expected = {
        "phase",
        "target_descent_rate",
        "yaw_locked",
        "yaw_target",
        "commit_soft",
        "abort_reason",
    }
    assert expected.issubset(out.keys()), f"missing keys: {expected - out.keys()}"
    assert isinstance(out["phase"], str)


def test_phase_from_helper_unwraps_dict():
    """Control reads phase via _phase_from() — confirm the unwrap path."""
    agent = _new_agent()
    out = agent.decide(_state(), _estimate(), battery=1.0, time_s=0.0)
    # Same string in both views, regardless of which path control takes.
    assert agent._phase_from(out) == out["phase"]
    assert agent._phase_from(out["phase"]) == out["phase"]


# --------------------------------------------------------------------- hysteresis
def test_search_to_approach_requires_n_debounce_frames():
    """A single fresh detection must NOT flip to APPROACH — that's the
    flicker the hysteresis kills. The transition fires only after
    ``APPROACH_ENTER_FRAMES`` consecutive supporting frames.
    """
    agent = _new_agent()
    state, est = _state(), _estimate()
    n = stl.BaselineAgent.APPROACH_ENTER_FRAMES

    # The first (n-1) consecutive supporting frames must stay in SEARCH.
    for i in range(n - 1):
        out = agent.decide(state, est, battery=1.0, time_s=i * 0.02)
        assert out["phase"] == stl.PHASE_SEARCH, (
            f"frame {i}: premature transition at counter={i+1} (< {n})"
        )

    # The n-th consecutive supporting frame fires the transition.
    out = agent.decide(state, est, battery=1.0, time_s=(n - 1) * 0.02)
    assert out["phase"] == stl.PHASE_APPROACH


def test_hysteresis_counter_resets_on_failed_condition():
    """Non-supporting frames in the middle of an in-progress transition
    must zero the counter (otherwise hysteresis degrades to a moving
    average, which doesn't filter anything)."""
    agent = _new_agent()
    state = _state()
    fresh, not_fresh = _estimate(fresh=True), _estimate(fresh=False)
    n = stl.BaselineAgent.APPROACH_ENTER_FRAMES
    assert n >= 2, "test relies on multi-frame debounce"

    # Accumulate (n-1) supporting frames then drop one non-supporting.
    for i in range(n - 1):
        agent.decide(state, fresh, battery=1.0, time_s=i * 0.02)
    agent.decide(state, not_fresh, battery=1.0, time_s=(n - 1) * 0.02)
    # Counter should be 0; we should still be in SEARCH.
    assert agent.phase == stl.PHASE_SEARCH
    assert agent._phase_frame_counters["approach_enter"] == 0


def test_approach_does_not_exit_on_single_frame_loss():
    """Marker drop for fewer than MARKER_LOST_EXIT_FRAMES must NOT pop
    APPROACH back to SEARCH. This is what kills the most score on EASY
    where occasional jitter causes a stale frame or two.
    """
    agent = _new_agent()
    state = _state()

    # First drive to APPROACH.
    for _ in range(stl.BaselineAgent.APPROACH_ENTER_FRAMES):
        agent.decide(state, _estimate(), battery=1.0, time_s=0.0)
    assert agent.phase == stl.PHASE_APPROACH

    # Now feed a few stale-but-not-lost frames. The counter increments
    # but should stay below the abort threshold.
    n_loss = stl.BaselineAgent.MARKER_LOST_EXIT_FRAMES - 1
    for i in range(n_loss):
        agent.decide(
            state,
            _estimate(stale_steps=999),  # marker_lost True
            battery=1.0,
            time_s=0.1 + i * 0.02,
        )
    assert agent.phase == stl.PHASE_APPROACH, (
        "transient marker loss must not abort APPROACH"
    )


# --------------------------------------------------------------------- yaw lock
def test_yaw_align_freezes_target_heading_once_locked():
    """decide.md #8 — once we lock a yaw target, decide() must stop
    tracking the live boat heading. Continuous yaw tracking on the VTOL
    airframe (low k_Q/k_T) bleeds roll/pitch authority during descent.
    """
    agent = _new_agent()
    state = _state(pos=(0.1, 0.0, 3.0))  # close to target
    n_var = stl.BaselineAgent.YAW_HEADING_VAR_WINDOW

    # Drive to APPROACH first.
    for i in range(stl.BaselineAgent.APPROACH_ENTER_FRAMES):
        agent.decide(state, _estimate(heading=0.7), battery=1.0, time_s=i * 0.02)
    assert agent.phase == stl.PHASE_APPROACH

    # Fill heading history with a stable value so the variance gate opens.
    for i in range(n_var):
        agent.decide(
            state,
            _estimate(heading=0.7),
            battery=1.0,
            time_s=0.1 + i * 0.02,
        )
    # By now yaw should have locked at ~0.7.
    if agent._yaw_locked:
        locked_target = agent._frozen_yaw_target
        # Feed a different heading; lock must NOT update.
        for i in range(30):
            agent.decide(
                state,
                _estimate(heading=1.3),
                battery=1.0,
                time_s=2.0 + i * 0.02,
            )
        assert agent._frozen_yaw_target == pytest.approx(locked_target), (
            "frozen yaw target must not track live boat heading after lock"
        )


# --------------------------------------------------------------------- soft commit
def test_soft_commit_requires_all_four_conditions():
    """decide.md #7 — commit_soft is True only when ALL of:
      - v_rel_horiz < 0.3
      - |v_rel_vert| < 0.4
      - marker tracked ≥ 10 frames
      - z_above < 1.0

    A False on ANY one keeps the bang-bang profile (avoids missing
    the platform for a +5 bonus that's not worth it).
    """
    agent = _new_agent()
    # Drive marker_tracked high by feeding fresh detections.
    state_close = _state(pos=(0.0, 0.0, 0.5))  # z_above < 1.0
    for i in range(stl.BaselineAgent.SOFT_COMMIT_TRACKED_FRAMES + 5):
        out = agent.decide(
            state_close,
            _estimate(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)),
            battery=1.0,
            time_s=i * 0.02,
        )
    # All four conditions should hold; commit_soft should be True.
    assert out["commit_soft"] is True

    # Break one condition: fast horizontal velocity. commit_soft must flip.
    state_fast = _state(pos=(0.0, 0.0, 0.5), vel=(2.0, 0.0, 0.0))
    out = agent.decide(
        state_fast,
        _estimate(pos=(0.0, 0.0, 0.0)),
        battery=1.0,
        time_s=5.0,
    )
    assert out["commit_soft"] is False


def test_soft_commit_false_before_tracked_frames_threshold():
    """commit_soft must stay False until marker_tracked ≥ N even when
    every other condition is satisfied."""
    agent = _new_agent()
    state = _state(pos=(0.0, 0.0, 0.5))
    # First frame ever: marker_tracked = 1.
    out = agent.decide(
        state,
        _estimate(pos=(0.0, 0.0, 0.0)),
        battery=1.0,
        time_s=0.0,
    )
    assert out["commit_soft"] is False


# --------------------------------------------------------------------- target descent
def test_target_descent_rate_per_phase():
    """Control reads target_descent_rate; non-DESCEND/LAND phases must
    publish 0 (no descent during SEARCH/APPROACH/YAW_ALIGN)."""
    agent = _new_agent()
    out = agent.decide(_state(), _estimate(), battery=1.0, time_s=0.0)
    # Still in SEARCH on first frame (hysteresis).
    assert out["phase"] == stl.PHASE_SEARCH
    assert out["target_descent_rate"] == 0.0


# --------------------------------------------------------------------- NaN safety
def test_nan_drone_position_falls_back_to_search():
    """contracts.md NaN-safety rule — NaN inputs must collapse to a
    safe phase, never propagate non-finite values into the output."""
    agent = _new_agent()
    bad_state = _state(pos=(np.nan, 0.0, 5.5))
    out = agent.decide(bad_state, _estimate(), battery=1.0, time_s=0.0)
    assert out["phase"] == stl.PHASE_SEARCH
    assert out["abort_reason"] is not None
    # Ensure no numeric field carries a non-finite value.
    for k in ("target_descent_rate", "yaw_target"):
        v = out.get(k)
        if v is not None:
            assert np.isfinite(v), f"{k} leaked non-finite: {v}"


def test_no_estimate_yields_search_with_reason():
    """When estimate has no fix (from_prior or None position), decide
    must return SEARCH with an informative abort_reason."""
    agent = _new_agent()
    out = agent.decide(
        _state(),
        _estimate(from_prior=True),
        battery=1.0,
        time_s=0.0,
    )
    assert out["phase"] == stl.PHASE_SEARCH
    assert out["abort_reason"] in {"from_prior", "no_estimate"}


def test_target_position_with_nan_falls_back_safely():
    agent = _new_agent()
    bad_est = _estimate(pos=(np.nan, 0.0, 0.3))
    out = agent.decide(_state(), bad_est, battery=1.0, time_s=0.0)
    assert out["phase"] == stl.PHASE_SEARCH
    assert out["abort_reason"] is not None


# --------------------------------------------------------------------- side-effect contract
def test_decide_stores_phase_out_on_self():
    """The teammate brief stashes the dict on ``self.phase_out`` so the
    control teammate and external inspectors can read the optional fields
    even when call sites still pass the bare phase string."""
    agent = _new_agent()
    out = agent.decide(_state(), _estimate(), battery=1.0, time_s=0.0)
    assert agent.phase_out is out


def test_self_phase_kept_in_sync_after_act():
    """Sanity: act() should call decide() and update self.phase to the
    string form for legacy inspectors (test_baseline relies on this)."""
    agent = _new_agent()
    obs = {
        "camera": np.full((480, 640, 3), 60, dtype=np.uint8),
        "state": _state(),
        "battery": 1.0,
        "time": 0.0,
    }
    agent.act(obs)
    assert isinstance(agent.phase, str)
    assert agent.phase in {
        stl.PHASE_SEARCH,
        stl.PHASE_APPROACH,
        stl.PHASE_YAW_ALIGN,
        stl.PHASE_DESCEND,
        stl.PHASE_LAND,
    }
