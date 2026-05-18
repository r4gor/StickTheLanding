"""Unit tests for the perception stage of agent.py.

Run inside Docker, from the repo root:
    ./docker/run-local.sh pytest teams/stick-the-landing/test_perception.py -v

Synthetic tests (no PyBullet env): detection, pose recovery, the camera->world
transform, the stale-frame flag, and the defog/deconv/CNN cascade tiers.
End-to-end accuracy against ground truth is covered by check_perception.py.
"""

import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from boat_landing.camera import (                                         # noqa: E402
    apply_fog,
    apply_motion_blur,
    apply_noise,
    get_intrinsics,
)
from agent import Agent, MARKER_SIZE_M, marker_to_world, rpy_to_matrix    # noqa: E402


def _synthetic_frame(marker_px: int = 240, frame_hw=(480, 640)) -> np.ndarray:
    """Render marker 0 fronto-parallel and centred, with a white quiet zone —
    the cleanest possible input for the detector."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    try:
        marker = cv2.aruco.generateImageMarker(adict, 0, marker_px)
    except AttributeError:  # OpenCV < 4.7
        marker = cv2.aruco.drawMarker(adict, 0, marker_px)

    pad = marker_px // 4
    tile = np.full((marker_px + 2 * pad, marker_px + 2 * pad), 255, np.uint8)
    tile[pad:pad + marker_px, pad:pad + marker_px] = marker

    h, w = frame_hw
    img = np.full((h, w), 255, np.uint8)
    th, tw = tile.shape
    y0, x0 = (h - th) // 2, (w - tw) // 2
    img[y0:y0 + th, x0:x0 + tw] = tile
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# --------------------------------------------------------------- perceive()

def test_perceive_blank_frame_reports_no_detection():
    out = Agent().perceive(np.full((480, 640, 3), 128, np.uint8))
    assert out["detected"] is False
    assert "stale" in out


def test_perceive_detects_synthetic_marker():
    out = Agent().perceive(_synthetic_frame(marker_px=240))
    assert out["detected"] is True
    assert out["reprojection_error"] < 2.0    # fronto-parallel -> tiny error
    assert out["tvec"].shape == (3,)
    assert out["rvec"].shape == (3,)
    assert out["image_points"].shape == (4, 2)
    assert out["source"] == "aruco"           # clean frame should hit the fast path


def test_perceive_depth_matches_pinhole_geometry():
    """For a fronto-parallel marker, depth = fx * size / observed_px."""
    out = Agent().perceive(_synthetic_frame(marker_px=240))
    assert out["detected"] is True
    fx = get_intrinsics()[0, 0]
    # Reconstruct the marker's observed pixel size from image_points (the
    # live perception contract no longer exposes a separate `marker_px`).
    ip = out["image_points"]
    side_px = float(np.mean([
        np.linalg.norm(ip[0] - ip[1]),
        np.linalg.norm(ip[1] - ip[2]),
        np.linalg.norm(ip[2] - ip[3]),
        np.linalg.norm(ip[3] - ip[0]),
    ]))
    expected_depth = fx * MARKER_SIZE_M / side_px
    assert abs(out["tvec"][2] - expected_depth) / expected_depth < 0.15
    # centred marker -> pose essentially on the optical axis
    assert abs(out["tvec"][0]) < 0.1 * out["tvec"][2]
    assert abs(out["tvec"][1]) < 0.1 * out["tvec"][2]


def test_stale_flag_set_on_repeated_frame():
    agent = Agent()
    frame = _synthetic_frame()
    first = agent.perceive(frame)
    second = agent.perceive(frame.copy())     # byte-identical -> stale
    assert first["stale"] is False
    assert second["stale"] is True


def test_distinct_frames_are_not_stale():
    agent = Agent()
    agent.perceive(_synthetic_frame(marker_px=240))
    out = agent.perceive(_synthetic_frame(marker_px=200))
    assert out["stale"] is False


# ---------------------------------------------------------------- cascade tiers

def test_cascade_defog_helps_on_fogged_frame():
    """Heavy Beer-Lambert fog washes contrast out. Dark-channel defog
    composed with CLAHE should recover the marker on the first ArUco pass."""
    frame = apply_fog(_synthetic_frame(marker_px=240), fog_density=0.10, altitude_m=5.0)
    out = Agent().perceive(frame)
    assert out["detected"] is True
    assert out["source"] == "aruco"   # defog (always-on) keeps us on the fast tier


def test_cascade_deconv_recovers_blurred_frame():
    """Horizontal motion blur breaks ArUco's grid decoding even when polygons
    are still visible. Wiener deconvolution fallback should re-decode."""
    frame = apply_motion_blur(_synthetic_frame(marker_px=240), kernel_size=3)
    out = Agent().perceive(frame)
    assert out["detected"] is True
    assert out["source"] == "deconv"  # ArUco missed; Wiener path recovered


def test_cascade_skips_wiener_when_no_polygon_candidates():
    """Without polygon candidates (n_rejected == 0) the Wiener retry must
    NOT run — its second adaptive-threshold scan would burn ~50 ms for
    nothing. Spy on _deblur_wiener and assert it isn't called."""
    agent = Agent()
    calls = {"wiener": 0}
    real_wiener = agent._deblur_wiener
    def spy(gray):
        calls["wiener"] += 1
        return real_wiener(gray)
    agent._deblur_wiener = spy  # type: ignore[assignment]

    rng = np.random.default_rng(0)
    flat = np.full((480, 640, 3), 180, np.uint8)
    fresh = np.clip(
        flat.astype(np.int16) + rng.integers(-3, 3, flat.shape), 0, 255
    ).astype(np.uint8)
    out = agent.perceive(fresh)
    assert out["detected"] is False
    assert calls["wiener"] == 0, "Wiener ran on a featureless frame (no candidates)"


def test_cascade_runs_wiener_when_polygon_candidates_present():
    """Blur-only marker frame: ArUco's first pass should see polygons
    (n_rejected > 0) but fail to decode marker 0; Wiener fallback then runs."""
    agent = Agent()
    calls = {"wiener": 0}
    real_wiener = agent._deblur_wiener
    def spy(gray):
        calls["wiener"] += 1
        return real_wiener(gray)
    agent._deblur_wiener = spy  # type: ignore[assignment]

    frame = apply_motion_blur(_synthetic_frame(marker_px=240), kernel_size=3)
    out = agent.perceive(frame)
    assert out["detected"] is True
    assert out["source"] == "deconv"
    assert calls["wiener"] == 1, "Wiener should have run exactly once on the fallback path"


def test_cascade_cnn_missing_does_not_raise():
    """When models/corner_detector.onnx is absent the CNN fallback must
    cleanly stay disabled and perceive() returns detected=False without
    raising on a frame ArUco cannot solve."""
    agent = Agent()
    # If the model file is committed in CI, this assertion is moot; otherwise
    # _cnn_ready should be False.
    if agent._cnn_ready:
        pytest.skip("corner_detector.onnx is present — CNN path active")
    bad = apply_noise(
        apply_motion_blur(
            apply_fog(_synthetic_frame(marker_px=240), 0.10, 5.0),
            5,
        ),
        0.06,
        np.random.default_rng(0),
    )
    out = agent.perceive(bad)
    # Either detected (defog+deconv recovered) or cleanly not detected;
    # the contract is just that no exception is raised and stale is set.
    assert "detected" in out and "stale" in out


def test_stale_short_circuit_returns_cached_detection():
    """Identical consecutive frames should return the cached perception dict
    with stale=True — avoids re-running defog/ArUco on byte-identical
    repeats (the cost driver on camera-fps-throttled scenarios)."""
    agent = Agent()
    frame = _synthetic_frame(240)
    first = agent.perceive(frame)
    second = agent.perceive(frame.copy())  # byte-identical -> stale
    assert second["stale"] is True
    # The cached dict carries the same detection result.
    assert second["detected"] == first["detected"]
    if first["detected"]:
        assert np.allclose(second["tvec"], first["tvec"])


# ----------------------------------------------------------- rpy_to_matrix()

def test_rpy_to_matrix_identity_at_zero():
    assert np.allclose(rpy_to_matrix([0.0, 0.0, 0.0]), np.eye(3))


def test_rpy_to_matrix_is_a_rotation():
    R = rpy_to_matrix([0.3, -0.2, 1.1])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


# --------------------------------------------------------- marker_to_world()

def test_marker_to_world_returns_none_without_detection():
    mw, yaw = marker_to_world(
        {"detected": False},
        {"position": np.zeros(3), "attitude": np.zeros(3)},
    )
    assert mw is None and yaw is None


def test_marker_to_world_level_drone_marker_on_optical_axis():
    """Level drone, marker straight down the optical axis: the marker shares
    the drone's x,y and sits tvec[2] below the camera (camera is 0.115 m below
    the COM)."""
    perception = {
        "detected": True,
        "tvec": np.array([0.0, 0.0, 5.0]),
        "rvec": np.zeros(3),
    }
    state = {"position": np.array([2.0, -1.0, 5.0]), "attitude": np.zeros(3)}
    mw, yaw = marker_to_world(perception, state)
    assert np.allclose(mw[:2], [2.0, -1.0], atol=1e-9)
    assert abs(mw[2] - (5.0 - 0.115 - 5.0)) < 1e-9
    assert np.isfinite(yaw)


def test_marker_to_world_end_to_end_with_real_detection():
    """perceive() on a synthetic frame -> marker_to_world: a level drone at
    altitude should place the marker x,y under the drone and z below it."""
    out = Agent().perceive(_synthetic_frame(marker_px=240))
    assert out["detected"] is True
    drone_z = 5.0
    state = {"position": np.array([3.0, 4.0, drone_z]), "attitude": np.zeros(3)}
    mw, yaw = marker_to_world(out, state)
    assert np.linalg.norm(mw[:2] - np.array([3.0, 4.0])) < 0.15
    assert mw[2] < drone_z
    assert np.isfinite(yaw)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
