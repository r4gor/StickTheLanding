"""Gymnasium wrapper for RL training on the Catch the Boat challenge.

Wraps BoatLandingEnv into a standard gymnasium.Env with:
    - 18-dim observation (from perceive/estimate contracts + drone state)
    - 4-dim continuous action (thrust_norm, roll, pitch, yaw_rate)
    - Dense reward shaped to align with AGENT_SCORING.md
    - Terminal reward using the exact scorer formula

The wrapper re-uses the baseline's perceive() and estimate() methods
so RL training can start immediately. When teammates upgrade those
pipelines, the wrapper picks up the improvements with no code changes.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Make the repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boat_landing.env import BoatLandingEnv
from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.drone_interface import DroneSpec, load_drone_spec
from evaluation.scorer import compute_score

import cv2


# ---------------------------------------------------------------------------
# Load the deployment BaselineAgent via importlib so RL training sees the
# SAME perception + estimation cascade the agent will run at evaluation.
# Avoids the stale-baseline RL/deployment mismatch.
# ---------------------------------------------------------------------------
def _load_baseline_agent_module():
    _agent_path = Path(__file__).resolve().parent / "agent.py"
    _spec = importlib.util.spec_from_file_location(
        "stl_baseline_agent_env", str(_agent_path)
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


# ---------------------------------------------------------------------------
# Helpers (copied from baseline to avoid circular imports)
# ---------------------------------------------------------------------------

MARKER_SIZE_M = 0.8


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
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_error_mod_pi(drone_yaw: float, boat_heading: float) -> float:
    """Compute yaw alignment error mod pi (both orientations are valid)."""
    err = _wrap_angle(drone_yaw - boat_heading)
    # Mod pi: both 0 and pi are valid alignments
    if abs(err) > np.pi / 2:
        err = _wrap_angle(err - np.pi)
    return err


# ---------------------------------------------------------------------------
# Perception + estimation delegators
#
# GAP 2 FIX: rather than maintaining a duplicate stale-baseline copy here,
# we delegate to teams/stick-the-landing/agent.py::BaselineAgent so RL
# training observes the SAME perception + estimation cascade that runs at
# evaluation time. A single BaselineAgent instance holds both stages
# (perceive is stateless modulo the frame-hash cache, estimate is stateful
# on a KF) — sharing the instance keeps the KF state coherent.
# ---------------------------------------------------------------------------

class _BaselineCascade:
    """Wraps a BaselineAgent instance to expose perceive() + estimate() with
    a reset() hook for the gym wrapper. Falls back to a minimal stale
    pipeline if BaselineAgent cannot be loaded so training never crashes.
    """

    SEARCH_PRIOR_XY = np.array([0.0, 0.0])

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        self._drone_spec = drone_spec
        self._main_agent = None
        self._mod = None
        try:
            self._mod = _load_baseline_agent_module()
            self._main_agent = self._mod.BaselineAgent(drone_spec=drone_spec)
            print(
                "[RL-int] [rl_env] Delegating perceive/estimate to main "
                "BaselineAgent (defog+CLAHE+IPPE+Wiener+CNN + KF).",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[RL-int] [rl_env] WARNING: failed to load main BaselineAgent "
                f"({exc}); using fallback stale baseline.",
                file=sys.stderr,
            )

        # Fallback path state — only used if BaselineAgent failed to load.
        self._fb_intrinsics = get_intrinsics()
        self._fb_dist = np.zeros(5, dtype=np.float64)
        self._fb_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_100
        )
        try:
            self._fb_detector = cv2.aruco.ArucoDetector(
                self._fb_dict, cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            self._fb_detector = None
        h = MARKER_SIZE_M / 2.0
        self._fb_object_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )
        self._fb_last_marker_world: Optional[np.ndarray] = None
        self._fb_frames_since_detection = 0

    def reset(self):
        """Reset stateful estimation between episodes.

        For the delegated path we rebuild a fresh BaselineAgent so the KF
        state, frame caches, and detection counters start clean. The cost
        is small relative to a training rollout.
        """
        if self._mod is not None:
            try:
                self._main_agent = self._mod.BaselineAgent(
                    drone_spec=self._drone_spec
                )
            except Exception as exc:
                print(
                    f"[RL-int] [rl_env] WARNING: BaselineAgent reset failed "
                    f"({exc}); using last instance.",
                    file=sys.stderr,
                )
        self._fb_last_marker_world = None
        self._fb_frames_since_detection = 0

    def perceive(self, camera_image: np.ndarray) -> Dict:
        if self._main_agent is not None:
            return self._main_agent.perceive(camera_image)
        return self._fallback_perceive(camera_image)

    def estimate(self, perception: Dict, drone_state: Dict) -> Dict:
        if self._main_agent is not None:
            return self._main_agent.estimate(perception, drone_state)
        return self._fallback_estimate(perception, drone_state)

    # ------------------------------------------------------------ fallback
    def _fallback_perceive(self, camera_image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        if self._fb_detector is not None:
            corners, ids, _ = self._fb_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._fb_dict)
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
            self._fb_object_points,
            image_points,
            self._fb_intrinsics,
            self._fb_dist,
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

    def _fallback_estimate(
        self, perception: Dict, drone_state: Dict
    ) -> Dict:
        if perception.get("detected", False):
            tvec = perception["tvec"]
            drone_pos = np.asarray(
                drone_state["position"], dtype=np.float64
            )
            R_world_body = _rpy_to_matrix(drone_state["attitude"])
            body_y = R_world_body[:, 1]
            body_z = R_world_body[:, 2]
            body_x = R_world_body[:, 0]
            camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec
            self._fb_last_marker_world = marker_world
            self._fb_frames_since_detection = 0
            return {
                "position": marker_world,
                "velocity": np.zeros(3),
                "fresh": True,
                "stale_steps": 0,
                "from_prior": False,
                "heading": None,
                "heading_rate": None,
                "confidence": None,
            }
        self._fb_frames_since_detection += 1
        if self._fb_last_marker_world is not None:
            return {
                "position": self._fb_last_marker_world,
                "velocity": np.zeros(3),
                "fresh": False,
                "stale_steps": self._fb_frames_since_detection,
                "from_prior": False,
                "heading": None,
                "heading_rate": None,
                "confidence": None,
            }
        return {
            "position": np.array(
                [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]
            ),
            "velocity": np.zeros(3),
            "fresh": False,
            "stale_steps": self._fb_frames_since_detection,
            "from_prior": True,
            "heading": None,
            "heading_rate": None,
            "confidence": None,
        }


# ---------------------------------------------------------------------------
# Main Gymnasium Wrapper
# ---------------------------------------------------------------------------

class BoatLandingGymWrapper(gym.Env):
    """Gymnasium environment wrapping BoatLandingEnv for RL training.

    Observation: 18-dim float32 vector (see _build_state).
    Action: 4-dim float32 (thrust_norm, roll, pitch, yaw_rate).
    Reward: dense per-step shaping + terminal scorer reward.
    """

    metadata = {"render_modes": ["human"]}

    # State vector dimension
    OBS_DIM = 18
    # Normalization constants
    MAX_HORIZ_DIST = 20.0
    MAX_VEL = 5.0
    MAX_ALT = 15.0
    MAX_STALE = 50  # frames — 1 second at 50 Hz
    MAX_DESCENT_VEL = 5.0

    def __init__(
        self,
        scenario: str = "easy",
        drone_sim=None,
        seed: Optional[int] = None,
        gui: bool = False,
    ):
        super().__init__()
        # Resolve scenario path
        scenario_path = REPO_ROOT / "scenarios" / f"{scenario}.yaml"
        if not scenario_path.is_file():
            scenario_path = Path(scenario)

        # Build the drone simulator if not provided
        if drone_sim is None:
            from agents.drone_sim_baseline import BaselineDroneSimulator
            drone_sim = BaselineDroneSimulator(
                str(REPO_ROOT / "drones" / "vtol.yaml")
            )

        self._env = BoatLandingEnv(
            str(scenario_path), drone_sim=drone_sim, gui=gui
        )
        self._scenario_name = scenario
        self._seed = seed
        self._duration_max = float(
            self._env.scenario.get("duration_max", 60.0)
        )

        # Perception + estimation pipelines — single delegated cascade so
        # perceive() and estimate() share the BaselineAgent's KF state.
        self._cascade = _BaselineCascade(drone_spec=self._env.drone_spec)
        # Back-compat aliases — anything that referenced _perception /
        # _estimation still works.
        self._perception = self._cascade
        self._estimation = self._cascade

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.ones(self.OBS_DIM, dtype=np.float32) * 2.0,
            high=np.ones(self.OBS_DIM, dtype=np.float32) * 2.0,
            dtype=np.float32,
        )
        # Action: (thrust_norm, roll, pitch, yaw_rate)
        # thrust_norm: [-1, 1], roll: [-0.5, 0.5], pitch: [-0.5, 0.5],
        # yaw_rate: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.5, -0.5, -1.0], dtype=np.float32),
            high=np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state tracking
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._max_descent_vel = 0.0
        self._last_obs: Optional[Dict] = None
        self._last_info: Optional[Dict] = None
        self._last_boat_estimate: Optional[Dict] = None
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # 1. Update/increment seed to ensure stochastic episodes (wind, noise, waves)
        if seed is not None:
            self._seed = seed
        elif self._seed is not None:
            self._seed = (self._seed + 1) % (2**32)
        else:
            self._seed = int(np.random.randint(0, 1000000))

        # 2. Domain Randomization: Perturb drone start position slightly on every reset
        # This forces the agent to learn a general control law rather than a specific trajectory.
        base_pos = np.asarray(self._env.scenario["drone_start"]["position"], dtype=np.float64)
        temp_rng = np.random.default_rng(self._seed)
        # Add random offset: up to +/- 1.5m in X and Y, +/- 0.5m in Z
        offset = temp_rng.uniform([-1.5, -1.5, -0.5], [1.5, 1.5, 0.5])
        perturbed_pos = base_pos + offset

        # Temporarily override scenario start position for this episode's reset
        original_pos = self._env.scenario["drone_start"]["position"]
        self._env.scenario["drone_start"]["position"] = perturbed_pos.tolist()

        # Reset the underlying env
        obs, info = self._env.reset(seed=self._seed)

        # Restore original scenario start position
        self._env.scenario["drone_start"]["position"] = original_pos

        # Reset internal state
        self._cascade.reset()
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._max_descent_vel = 0.0
        self._step_count = 0

        # Run perception + estimation on the first frame
        percept = self._cascade.perceive(obs["camera"])
        boat_est = self._cascade.estimate(percept, obs["state"])

        self._last_obs = obs
        self._last_info = info
        self._last_boat_estimate = boat_est

        state_vec = self._build_state(obs, boat_est)
        return state_vec, {"obs": obs, "info": info, "boat_estimate": boat_est}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(4)

        # Clamp action to valid ranges
        action = np.clip(action, self.action_space.low, self.action_space.high)

        thrust_norm = float(action[0])
        roll_des = float(action[1])
        pitch_des = float(action[2])
        yaw_rate_des = float(action[3])

        # Convert setpoints to motor throttles via the stock attitude ctrl
        from boat_landing.controllers import DefaultAttitudeController
        if not hasattr(self, "_attitude_ctrl"):
            self._attitude_ctrl = DefaultAttitudeController(
                self._env.drone_spec
            )

        motor_action = self._attitude_ctrl(
            self._last_obs["state"],
            roll_des=roll_des,
            pitch_des=pitch_des,
            yaw_rate_des=yaw_rate_des,
            thrust_norm=thrust_norm,
        )

        # Step the underlying env
        obs, _, terminated, truncated, info = self._env.step(motor_action)

        # Run perception + estimation
        percept = self._cascade.perceive(obs["camera"])
        boat_est = self._cascade.estimate(percept, obs["state"])

        # Track max descent velocity
        vz = float(obs["state"]["velocity"][2])
        if -vz > self._max_descent_vel:
            self._max_descent_vel = -vz

        # Compute reward
        reward = self._compute_reward(
            obs, info, boat_est, action, terminated, truncated
        )

        # Build next state
        state_vec = self._build_state(obs, boat_est)

        self._last_obs = obs
        self._last_info = info
        self._last_boat_estimate = boat_est
        self._prev_action = action.copy()
        self._step_count += 1

        return state_vec, reward, terminated, truncated, {
            "obs": obs,
            "info": info,
            "boat_estimate": boat_est,
        }

    def close(self):
        self._env.close()

    # --------------------------------------------------------------- state
    def _build_state(
        self, obs: Dict, boat_est: Dict
    ) -> np.ndarray:
        """Build the 18-dim normalized state vector.

        Dims 0-2:  relative position (body frame), normalized
        Dims 3-5:  relative velocity (body frame), normalized
        Dims 6-7:  roll, pitch
        Dim  8:    yaw alignment error (mod pi)
        Dim  9:    yaw rate, normalized
        Dims 10-11: boat velocity XY (world), normalized
        Dim  12:   estimation confidence
        Dim  13:   detection staleness, normalized
        Dim  14:   battery remaining
        Dim  15:   time remaining, normalized
        Dim  16:   altitude above boat, normalized
        Dim  17:   max descent velocity so far, normalized
        """
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

        # Rotation matrix (world from body)
        R = _rpy_to_matrix(drone_rpy)

        # Relative position in body frame (target is boat)
        rel_world = boat_pos - drone_pos
        rel_body = R.T @ rel_world

        # Relative velocity in body frame
        rel_vel_world = boat_vel - drone_vel
        rel_vel_body = R.T @ rel_vel_world

        # Yaw alignment error
        drone_yaw = float(drone_rpy[2])
        boat_heading = boat_est.get("heading")
        if boat_heading is not None:
            yaw_err = _yaw_error_mod_pi(drone_yaw, float(boat_heading))
        else:
            # No heading estimate yet — assume boat heading = 0 as fallback
            yaw_err = _yaw_error_mod_pi(drone_yaw, 0.0)

        # Estimation confidence
        confidence = boat_est.get("confidence")
        if confidence is None:
            # Heuristic: fresh detection = high confidence, stale = low
            stale = boat_est.get("stale_steps", 0)
            if boat_est.get("from_prior", False):
                confidence = 0.1
            elif stale == 0:
                confidence = 0.9
            else:
                confidence = max(0.1, 0.9 - stale * 0.05)

        # Staleness
        stale_steps = boat_est.get("stale_steps", 0)
        stale_norm = min(stale_steps / self.MAX_STALE, 1.0)

        # Time remaining
        time_remaining = max(
            0.0, self._duration_max - obs["time"]
        ) / self._duration_max

        # Altitude above boat
        z_above = drone_pos[2] - boat_pos[2]

        # Assemble and normalize
        state_vec = np.array([
            # 0-2: relative position (body), normalized
            np.clip(rel_body[0] / self.MAX_HORIZ_DIST, -1.0, 1.0),
            np.clip(rel_body[1] / self.MAX_HORIZ_DIST, -1.0, 1.0),
            np.clip(rel_body[2] / self.MAX_ALT, -1.0, 1.0),
            # 3-5: relative velocity (body), normalized
            np.clip(rel_vel_body[0] / self.MAX_VEL, -1.0, 1.0),
            np.clip(rel_vel_body[1] / self.MAX_VEL, -1.0, 1.0),
            np.clip(rel_vel_body[2] / self.MAX_VEL, -1.0, 1.0),
            # 6-7: roll, pitch (already small radians)
            np.clip(float(drone_rpy[0]) / 0.5, -1.0, 1.0),
            np.clip(float(drone_rpy[1]) / 0.5, -1.0, 1.0),
            # 8: yaw error mod pi, normalized to [-1, 1]
            np.clip(yaw_err / (np.pi / 2), -1.0, 1.0),
            # 9: yaw rate, normalized
            np.clip(float(drone_omega[2]) / 2.0, -1.0, 1.0),
            # 10-11: boat velocity XY (world), normalized
            np.clip(float(boat_vel[0]) / 3.0, -1.0, 1.0),
            np.clip(float(boat_vel[1]) / 3.0, -1.0, 1.0),
            # 12: estimation confidence
            float(confidence),
            # 13: detection staleness
            float(stale_norm),
            # 14: battery
            float(obs["battery"]),
            # 15: time remaining
            float(time_remaining),
            # 16: altitude above boat, normalized
            np.clip(z_above / self.MAX_ALT, -0.5, 1.0),
            # 17: max descent velocity so far, normalized
            np.clip(self._max_descent_vel / self.MAX_DESCENT_VEL, 0.0, 1.0),
        ], dtype=np.float32)

        return state_vec

    # --------------------------------------------------------------- reward
    def _compute_reward(
        self,
        obs: Dict,
        info: Dict,
        boat_est: Dict,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Dense per-step reward + terminal reward aligned with scorer."""
        state = obs["state"]
        drone_pos = np.asarray(state["position"], dtype=np.float64)
        drone_vel = np.asarray(state["velocity"], dtype=np.float64)
        drone_rpy = np.asarray(state["attitude"], dtype=np.float64)
        boat_pos = np.asarray(boat_est["position"], dtype=np.float64)

        # Horizontal distance to boat
        horiz_dist = float(
            np.linalg.norm(drone_pos[:2] - boat_pos[:2])
        )
        # Altitude above boat
        z_above = float(drone_pos[2] - boat_pos[2])

        # Yaw alignment error
        drone_yaw = float(drone_rpy[2])
        boat_heading = boat_est.get("heading")
        if boat_heading is not None:
            yaw_err = abs(_yaw_error_mod_pi(drone_yaw, float(boat_heading)))
        else:
            yaw_err = abs(_yaw_error_mod_pi(drone_yaw, 0.0))

        # Descent velocity (positive = going down)
        vz = -float(drone_vel[2])

        # ---- Dense reward components ----

        # 1. Approach: penalize horizontal distance
        r_approach = -0.1 * horiz_dist

        # 2. Altitude shaping
        if horiz_dist < 1.5:
            # When close horizontally, encourage descending
            # Target altitude decreases as we get closer
            target_z = max(0.3, horiz_dist * 2.0)
            r_altitude = -0.05 * abs(z_above - target_z)
        else:
            # When far, maintain a sensible search/approach altitude
            r_altitude = -0.02 * abs(z_above - 4.0)

        # 3. Yaw alignment pressure
        r_yaw = -0.15 * yaw_err

        # 4. Soft landing: penalize high descent speed near the deck
        r_soft = 0.0
        if z_above < 2.0 and vz > 0.8:
            r_soft = -0.3 * (vz - 0.8) ** 2

        # 5. Control smoothness
        r_smooth = -0.01 * float(
            np.sum((action - self._prev_action) ** 2)
        )

        # 6. Time penalty
        r_time = -0.002

        # 7. Alive bonus for being close to the boat (progress reward)
        if horiz_dist < 0.5 and z_above < 1.0:
            r_proximity = 0.05  # reward for being in landing position
        elif horiz_dist < 1.0:
            r_proximity = 0.02
        else:
            r_proximity = 0.0

        # 8. Estimation-bonus shaping (GAP 3): nudges the policy toward
        # behavior that keeps a fresh, low-uncertainty boat estimate, which
        # the scorer rewards via its RMSE bonus (0..15 pts). When the
        # delegated KF exposes a covariance we use it directly; otherwise
        # we use a fresh/stale proxy so the shaping is still active under
        # the fallback baseline.
        pos_cov = boat_est.get("position_cov")
        if pos_cov is not None:
            cov_arr = np.asarray(pos_cov, dtype=np.float64)
            if cov_arr.ndim == 2 and cov_arr.shape == (3, 3) and np.all(
                np.isfinite(cov_arr)
            ):
                r_est = -0.05 * float(np.trace(cov_arr))
            else:
                r_est = 0.0
        else:
            r_est = -0.05 * (0.0 if boat_est.get("fresh", False) else 10.0)

        dense_reward = (
            r_approach + r_altitude + r_yaw + r_soft
            + r_smooth + r_time + r_proximity + r_est
        )

        # ---- Terminal reward ----
        terminal_reward = 0.0
        if terminated or truncated:
            outcome = info.get("outcome")
            if outcome == "LANDED":
                # Use the exact scorer formula
                landing_pos_error = float(
                    np.linalg.norm(drone_pos[:2] - info["boat_position"][:2])
                )
                precision_factor = max(0.0, 1.0 - landing_pos_error / 1.5)
                time_factor = max(
                    0.5, 1.0 - obs["time"] / self._duration_max
                )
                battery_factor = max(0.3, float(obs["battery"]))

                terminal_reward = (
                    50.0 * precision_factor * time_factor * battery_factor
                )
                # Soft landing bonus
                if self._max_descent_vel < 1.0:
                    terminal_reward += 5.0
            elif outcome == "CRASHED":
                terminal_reward = -20.0
            elif outcome in ("TIMEOUT", "OUT_OF_BATTERY"):
                terminal_reward = -5.0  # mild penalty
            else:
                terminal_reward = -2.0  # other failures

        return dense_reward + terminal_reward
