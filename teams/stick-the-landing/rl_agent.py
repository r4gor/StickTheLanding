"""RL-based agent for the Catch the Boat challenge.

This agent replaces the baseline's hardcoded FSM + PID with a trained
neural network policy that outputs (thrust_norm, roll, pitch, yaw_rate)
setpoints. It exposes act_setpoint() for the +15 FC-compat bonus.

The perceive() and estimate() pipelines are kept from the baseline
(or upgraded by teammates). The RL policy only replaces decide()+control().

Usage:
    python evaluation/evaluate.py \\
        --agent teams/stick-the-landing/rl_agent.py \\
        --scenario easy --headless --seed 42
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Make the repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.drone_interface import DroneSpec, load_drone_spec

import cv2


# ---------------------------------------------------------------------------
# Load the deployment BaselineAgent (main's perceive + estimate cascade) via
# importlib so we delegate to the SAME perception+estimation that runs at
# evaluation time. Avoids the stale-baseline RL/deployment mismatch.
# ---------------------------------------------------------------------------
def _load_baseline_agent_module():
    _agent_path = Path(__file__).resolve().parent / "agent.py"
    _spec = importlib.util.spec_from_file_location(
        "stl_baseline_agent", str(_agent_path)
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


# ---------------------------------------------------------------------------
# Helpers (same as rl_env.py)
# ---------------------------------------------------------------------------

MARKER_SIZE_M = 0.8
MODEL_DIR = Path(__file__).resolve().parent / "models"


def _rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_error_mod_pi(drone_yaw: float, boat_heading: float) -> float:
    err = _wrap_angle(drone_yaw - boat_heading)
    if abs(err) > np.pi / 2:
        err = _wrap_angle(err - np.pi)
    return err


# ---------------------------------------------------------------------------
# RL Agent
# ---------------------------------------------------------------------------

class RLAgent:
    """RL-based landing agent.

    Exposes:
        act_setpoint(obs) -> (thrust, roll, pitch, yaw_rate)  [+15 FC bonus]
        get_last_estimate() -> dict                            [+15 est bonus]
    """

    # Normalization constants (must match rl_env.py)
    MAX_HORIZ_DIST = 20.0
    MAX_VEL = 5.0
    MAX_ALT = 15.0
    MAX_STALE = 50
    MAX_DESCENT_VEL = 5.0
    OBS_DIM = 18

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec = drone_spec

        # Try to load the trained model
        self._model = None
        model_path = MODEL_DIR / "landing_policy.zip"
        if model_path.exists():
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(str(model_path))
                print(
                    f"[RL-int] [RLAgent] Loaded trained model from {model_path}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"[RL-int] [RLAgent] WARNING: Failed to load model: {exc}. "
                    f"Using fallback hover policy.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[RL-int] [RLAgent] WARNING: No model found at {model_path}. "
                f"Using fallback hover policy. Train with train_rl.py first.",
                file=sys.stderr,
            )

        # GAP 1 FIX: delegate perceive() and estimate() to the deployment
        # BaselineAgent (defog + CLAHE + IPPE + Wiener + CNN cascade and the
        # KF estimator). Loaded via importlib to avoid a top-level circular
        # import. Any improvement to teams/stick-the-landing/agent.py is now
        # automatically picked up by RL training.
        try:
            _mod = _load_baseline_agent_module()
            self._main_agent = _mod.BaselineAgent(drone_spec=self.spec)
            print(
                "[RL-int] [RLAgent] Delegating perceive/estimate to main "
                "BaselineAgent (defog+CLAHE+IPPE+Wiener+CNN + KF).",
                file=sys.stderr,
            )
        except Exception as exc:
            self._main_agent = None
            print(
                f"[RL-int] [RLAgent] WARNING: Failed to load main BaselineAgent "
                f"({exc}); falling back to internal stale baseline.",
                file=sys.stderr,
            )

        # Fallback perception pipeline (only used if BaselineAgent failed to
        # load). Mirrors the stale baseline so the agent still runs in that
        # degraded case rather than crashing.
        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self._dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_100
        )
        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary, cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            self._detector = None

        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )

        # Estimation state (fallback path only)
        self._last_marker_world: Optional[np.ndarray] = None
        self._frames_since_detection = 0
        self._last_estimate: Optional[Dict] = None

        # RL state tracking
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._max_descent_vel = 0.0
        self._duration_max = 60.0  # default, updated on first obs

    # ---------------------------------------------------------------- public API

    def act_setpoint(self, obs: Dict) -> Tuple[float, float, float, float]:
        """FC-compatible output: (thrust_norm, roll, pitch, yaw_rate).

        Detected by evaluate.py → +15 FC-compat bonus.
        """
        # Run perception + estimation
        percept = self.perceive(obs["camera"])
        boat_est = self.estimate(percept, obs["state"])
        self._last_estimate = boat_est

        # Track descent velocity
        vz = -float(obs["state"]["velocity"][2])
        if vz > self._max_descent_vel:
            self._max_descent_vel = vz

        # Build state vector
        state_vec = self._build_state(obs, boat_est)

        # Get action from trained model (or fallback)
        if self._model is not None:
            action, _ = self._model.predict(state_vec, deterministic=True)
        else:
            # Fallback: gentle hover descending toward target
            action = self._fallback_action(obs, boat_est)

        action = np.asarray(action, dtype=np.float32).reshape(4)

        # Clamp to valid ranges
        thrust = float(np.clip(action[0], -1.0, 1.0))
        roll = float(np.clip(action[1], -0.5, 0.5))
        pitch = float(np.clip(action[2], -0.5, 0.5))
        yaw_rate = float(np.clip(action[3], -1.0, 1.0))

        # NaN guard
        if not all(np.isfinite([thrust, roll, pitch, yaw_rate])):
            thrust, roll, pitch, yaw_rate = 0.0, 0.0, 0.0, 0.0

        self._prev_action = np.array(
            [thrust, roll, pitch, yaw_rate], dtype=np.float32
        )

        return (thrust, roll, pitch, yaw_rate)

    def get_last_estimate(self) -> Optional[Dict]:
        """Expose boat estimate for the scorer's RMSE bonus."""
        if (
            self._last_estimate is None
            or self._last_estimate.get("position") is None
        ):
            return None
        return {
            "position": np.asarray(self._last_estimate["position"]),
            "velocity": np.asarray(
                self._last_estimate.get("velocity", np.zeros(3))
            ),
        }

    # ---------------------------------------------------------------- perception
    def perceive(self, camera_image: np.ndarray) -> Dict:
        """Delegate to deployment BaselineAgent.perceive (defog + CLAHE +
        IPPE + Wiener + CNN cascade). Falls back to a minimal ArUco-only
        implementation if BaselineAgent failed to load.
        """
        if self._main_agent is not None:
            return self._main_agent.perceive(camera_image)
        # Fallback path: minimal ArUco only.
        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary
            )

        if ids is None or len(ids) == 0:
            return {"detected": False}

        target = None
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                target = c
                break
        if target is None:
            return {"detected": False}

        image_points = target.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            self.intrinsics,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return {"detected": False}

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False}

        return {
            "detected": True,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
        }

    # ---------------------------------------------------------------- estimation
    def estimate(self, perception: Dict, drone_state: Dict) -> Dict:
        """Delegate to deployment BaselineAgent.estimate (KF on x,y,vx,vy +
        heading filter). Falls back to last-detected position if
        BaselineAgent failed to load.
        """
        if self._main_agent is not None:
            return self._main_agent.estimate(perception, drone_state)
        # Fallback path: stale baseline (no KF, no velocity).
        if perception.get("detected", False):
            tvec = perception["tvec"]
            drone_pos = np.asarray(
                drone_state["position"], dtype=np.float64
            )
            R_world_body = _rpy_to_matrix(drone_state["attitude"])
            body_y = R_world_body[:, 1]
            body_x = R_world_body[:, 0]
            body_z = R_world_body[:, 2]
            camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec
            self._last_marker_world = marker_world
            self._frames_since_detection = 0
            return {
                "position": marker_world,
                "velocity": np.zeros(3),
                "fresh": True,
                "stale_steps": 0,
                "from_prior": False,
                "heading": None,
                "confidence": None,
            }

        self._frames_since_detection += 1
        if self._last_marker_world is not None:
            return {
                "position": self._last_marker_world,
                "velocity": np.zeros(3),
                "fresh": False,
                "stale_steps": self._frames_since_detection,
                "from_prior": False,
                "heading": None,
                "confidence": None,
            }
        return {
            "position": np.array([0.0, 0.0, 0.3]),
            "velocity": np.zeros(3),
            "fresh": False,
            "stale_steps": self._frames_since_detection,
            "from_prior": True,
            "heading": None,
            "confidence": None,
        }

    # ---------------------------------------------------------------- state builder
    def _build_state(self, obs: Dict, boat_est: Dict) -> np.ndarray:
        """Build 18-dim normalized state vector (must match rl_env.py)."""
        state = obs["state"]
        drone_pos = np.asarray(state["position"], dtype=np.float64)
        drone_vel = np.asarray(state["velocity"], dtype=np.float64)
        drone_rpy = np.asarray(state["attitude"], dtype=np.float64)
        drone_omega = np.asarray(
            state["angular_velocity"], dtype=np.float64
        )

        boat_pos = np.asarray(boat_est["position"], dtype=np.float64)
        boat_vel = np.asarray(
            boat_est.get("velocity", np.zeros(3)), dtype=np.float64
        )

        R = _rpy_to_matrix(drone_rpy)
        rel_world = boat_pos - drone_pos
        rel_body = R.T @ rel_world
        rel_vel_world = boat_vel - drone_vel
        rel_vel_body = R.T @ rel_vel_world

        drone_yaw = float(drone_rpy[2])
        boat_heading = boat_est.get("heading")
        if boat_heading is not None:
            yaw_err = _yaw_error_mod_pi(drone_yaw, float(boat_heading))
        else:
            yaw_err = _yaw_error_mod_pi(drone_yaw, 0.0)

        confidence = boat_est.get("confidence")
        if confidence is None:
            stale = boat_est.get("stale_steps", 0)
            if boat_est.get("from_prior", False):
                confidence = 0.1
            elif stale == 0:
                confidence = 0.9
            else:
                confidence = max(0.1, 0.9 - stale * 0.05)

        stale_steps = boat_est.get("stale_steps", 0)
        stale_norm = min(stale_steps / self.MAX_STALE, 1.0)

        time_remaining = max(
            0.0, self._duration_max - obs["time"]
        ) / self._duration_max

        z_above = drone_pos[2] - boat_pos[2]

        return np.array([
            np.clip(rel_body[0] / self.MAX_HORIZ_DIST, -1.0, 1.0),
            np.clip(rel_body[1] / self.MAX_HORIZ_DIST, -1.0, 1.0),
            np.clip(rel_body[2] / self.MAX_ALT, -1.0, 1.0),
            np.clip(rel_vel_body[0] / self.MAX_VEL, -1.0, 1.0),
            np.clip(rel_vel_body[1] / self.MAX_VEL, -1.0, 1.0),
            np.clip(rel_vel_body[2] / self.MAX_VEL, -1.0, 1.0),
            np.clip(float(drone_rpy[0]) / 0.5, -1.0, 1.0),
            np.clip(float(drone_rpy[1]) / 0.5, -1.0, 1.0),
            np.clip(yaw_err / (np.pi / 2), -1.0, 1.0),
            np.clip(float(drone_omega[2]) / 2.0, -1.0, 1.0),
            np.clip(float(boat_vel[0]) / 3.0, -1.0, 1.0),
            np.clip(float(boat_vel[1]) / 3.0, -1.0, 1.0),
            float(confidence),
            float(stale_norm),
            float(obs["battery"]),
            float(time_remaining),
            np.clip(z_above / self.MAX_ALT, -0.5, 1.0),
            np.clip(
                self._max_descent_vel / self.MAX_DESCENT_VEL, 0.0, 1.0
            ),
        ], dtype=np.float32)

    # ---------------------------------------------------------------- fallback
    def _fallback_action(
        self, obs: Dict, boat_est: Dict
    ) -> np.ndarray:
        """Simple PD controller fallback when no model is loaded."""
        state = obs["state"]
        drone_pos = np.asarray(state["position"], dtype=np.float64)
        drone_vel = np.asarray(state["velocity"], dtype=np.float64)
        boat_pos = np.asarray(boat_est["position"], dtype=np.float64)

        R = _rpy_to_matrix(state["attitude"])
        err_world = boat_pos - drone_pos
        err_world[2] = (boat_pos[2] + 3.0) - drone_pos[2]  # hover 3m above
        err_body = R.T @ err_world
        vel_body = R.T @ drone_vel

        Kp, Kd = 0.15, 0.40
        pitch_cmd = np.clip(Kp * err_body[0] - Kd * vel_body[0], -0.3, 0.3)
        roll_cmd = np.clip(
            -(Kp * err_body[1] - Kd * vel_body[1]), -0.3, 0.3
        )
        thrust_cmd = np.clip(
            0.1 * err_world[2] - 0.3 * drone_vel[2], -1.0, 1.0
        )

        return np.array(
            [thrust_cmd, roll_cmd, pitch_cmd, 0.0], dtype=np.float32
        )


# ---------------------------------------------------------------- module API

# evaluation/evaluate.py looks for `make_agent` first, then `Agent`.
Agent = RLAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> RLAgent:
    return RLAgent(drone_spec)
