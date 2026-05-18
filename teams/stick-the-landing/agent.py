"""Baseline agent for the Catch the Boat challenge.

What this agent does:
    1. Detect the ArUco marker with cv2.aruco.ArucoDetector.
    2. Estimate marker pose with cv2.solvePnP (IPPE_SQUARE).
    3. Convert the marker's camera-frame translation to a world-frame
       position using the drone's attitude.
    4. Run a small state machine: SEARCH -> APPROACH -> DESCEND -> LAND.
    5. Drive three independent PIDs (x, y, z) to produce high-level
       (thrust, roll, pitch, yaw_rate) setpoints.
    6. Map those setpoints to per-motor throttles via the stock
       `DefaultAttitudeController`. The env's `step()` consumes
       per-motor throttles, not high-level setpoints.

If you want to write your own attitude controller (and earn the control-
fidelity portion of the sim-quality bonus), bypass step 6 and emit motor
throttles directly from `control()`.

Deliberate weaknesses (improvement directions for teams):
    - No prediction of boat motion: we always aim at the *current* marker
      pose, so we lag behind a moving boat.
    - No detection-jitter filter (no Kalman, no smoothing): every
      detection is trusted as-is.
    - No real recovery when the marker is briefly lost during descent.
      We hold the last known position and hope it returns. With
      occlusions, this leads to drift.
    - Hardcoded thresholds for state transitions and a hardcoded prior
      (origin) for cold-start search.
    - No abort/retry logic if the descent is going wrong (e.g. drone
      heading off the platform).
    - Independent x/y/z PIDs ignore the cross-coupling caused by tilt.
    - No active yaw control to align with boat heading — the wing-
      perpendicular landing condition will fail unless the boat happens
      to be aligned with the world x-axis at touchdown.
    - LAND-phase descent is bang-bang (thrust pinned to its lower bound)
      to brute-force through ground effect. Descent velocity often
      saturates and forfeits the soft-landing bonus. A smooth descent
      profile + velocity feed-forward both recover and stack.

Don't optimize the baseline. Fork agent_template.py and build your own.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parents[2]


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"


def rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation matrix using PyBullet's RPY convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Used to transform the marker's
    camera-frame translation into world coordinates."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _wrap_mod_pi(angle: float) -> float:
    """Wrap an angle into (-pi/2, pi/2]. The landing condition is fuselage
    alignment mod pi (both 0 and pi pass), so heading lives in a half-revolution
    range — this is the canonical representative."""
    return ((float(angle) + np.pi / 2.0) % np.pi) - np.pi / 2.0


class _BoatKalmanFilter:
    """Linear 4-state Kalman filter on world-frame ``[x, y, vx, vy]``.

    Constant-velocity (CV) process model with white-noise acceleration
    process noise (``sigma_a`` m/s^2 1-sigma per axis). Matches PX4's
    landing_target_estimator design and the team's estimate.md playbook.
    A single conservative ``sigma_a = 2.0`` covers EASY/MEDIUM/HARD —
    HARD's curve at 1.8 m/s with r=30 m has centripetal accel ~0.11 m/s^2,
    well inside the absorption band.
    """

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    def __init__(self, sigma_a: float = 2.0) -> None:
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0e6  # huge prior — uninitialised
        self.initialized = False
        self.sigma_a = float(sigma_a)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _Q(self, dt: float) -> np.ndarray:
        s = self.sigma_a ** 2
        return s * np.array(
            [[dt**4 / 4.0, 0.0,         dt**3 / 2.0, 0.0],
             [0.0,         dt**4 / 4.0, 0.0,         dt**3 / 2.0],
             [dt**3 / 2.0, 0.0,         dt**2,       0.0],
             [0.0,         dt**3 / 2.0, 0.0,         dt**2]],
            dtype=np.float64,
        )

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def innovation(self, z: np.ndarray, R: np.ndarray):
        """Return ``(innov, S)`` for outlier gating before committing."""
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        return innov, S

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        innov, S = self.innovation(z, R)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def initialize(self, z: np.ndarray, sigma_xy: float) -> None:
        self.x = np.array([float(z[0]), float(z[1]), 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([sigma_xy ** 2, sigma_xy ** 2, 1.0, 1.0]).astype(np.float64)
        self.initialized = True


class _HeadingKalmanFilter:
    """Scalar Kalman filter on ``[psi, psi_dot]`` for boat heading mod pi.

    Measurements come from ``cv2.Rodrigues(rvec)``; the mod-pi wrap lives
    in the innovation so the filter never sees a full-revolution
    discontinuity. Updates are gated by the caller (tilt > 15 deg, plus
    optional confidence > 0.5 when ``perceive()`` provides one)."""

    H = np.array([[1.0, 0.0]], dtype=np.float64)

    def __init__(
        self, sigma_psi_meas: float = 0.05, sigma_psi_dot: float = 0.3,
    ) -> None:
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0e6
        self.initialized = False
        self.sigma_psi_meas = float(sigma_psi_meas)
        self.sigma_psi_dot = float(sigma_psi_dot)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

    def _Q(self, dt: float) -> np.ndarray:
        s = self.sigma_psi_dot ** 2
        return s * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=np.float64,
        )

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.x[0] = _wrap_mod_pi(self.x[0])

    def update(self, psi_meas: float) -> None:
        if not self.initialized:
            return
        innov = _wrap_mod_pi(float(psi_meas) - float(self.x[0]))
        R = np.array([[self.sigma_psi_meas ** 2]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + (K.flatten() * innov)
        self.x[0] = _wrap_mod_pi(self.x[0])
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def initialize(self, psi_meas: float) -> None:
        self.x = np.array([_wrap_mod_pi(psi_meas), 0.0], dtype=np.float64)
        self.P = np.diag([self.sigma_psi_meas ** 2, 1.0]).astype(np.float64)
        self.initialized = True


class PID:
    """Single-axis PID with output clamping.

    Supports both 'D on error' (the textbook form) and 'D on measurement'
    (the form most controllers in robotics actually use). Pass
    ``measurement_velocity`` to ``__call__`` to use D on measurement —
    that avoids the derivative kick that hits when the setpoint jumps
    (e.g. when our boat estimate snaps to a fresh ArUco detection).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
        i_clip: float = 1.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_clip = i_clip
        self.reset()

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err: Optional[float] = None

    def __call__(
        self,
        err: float,
        dt: float,
        measurement_velocity: Optional[float] = None,
    ) -> float:
        self._i = float(np.clip(self._i + err * dt, -self.i_clip, self.i_clip))
        if measurement_velocity is not None:
            # err = setpoint - measurement, so d(err)/dt = -d(meas)/dt.
            d = -float(measurement_velocity)
        elif self._prev_err is None:
            d = 0.0
        else:
            d = (err - self._prev_err) / max(dt, 1e-6)
        self._prev_err = err
        out = self.kp * err + self.ki * self._i + self.kd * d
        return float(np.clip(out, self.out_min, self.out_max))


class BaselineAgent:
    """ArUco + state machine + 3 independent PIDs."""

    DT = 0.02  # must match BoatLandingEnv.DT

    # Position-KF outlier gating. 13.8 = chi^2 99.9% with 2 DoF — the team
    # raised this above the textbook 9.21 (99%) because gated NIS is not
    # exactly chi^2 distributed under heavy occlusion (estimate.md cites
    # arXiv 2512.18508). Five consecutive rejects trip a reinit-from-latest.
    NIS_THRESHOLD = 13.8
    NIS_REINIT_AFTER = 5

    # Trust thresholds. The position filter is "trusted" (`fresh=True`)
    # only after 3 consecutive gated updates — single-detection jitter no
    # longer flips `fresh` on its own.
    KF_TRUST_AFTER = 3

    # Stale-detection handling.
    STALE_INFLATE_AFTER_S = 1.5
    STALE_RESET_AFTER_S = 5.0

    # Heading filter gating. Drone-frame tilt > 15 deg means we're looking
    # at the marker from an angle, which disambiguates IPPE_SQUARE's
    # mirror pose at the rvec level.
    HEADING_TILT_GATE_RAD = np.deg2rad(15.0)
    HEADING_CONFIDENCE_GATE = 0.5
    HEADING_FLIP_REJECT_RAD = np.deg2rad(90.0)

    # Search prior: when the marker has never been detected, head toward
    # this world position at this altitude. The boats spawn near the
    # origin in all public scenarios.
    SEARCH_PRIOR_XY = np.array([0.0, 0.0])
    SEARCH_PRIOR_ALT = 5.5

    # Phase transition thresholds (m)
    APPROACH_HORIZ_OK = 0.6
    APPROACH_ALTITUDE_OK = 3.5
    DESCEND_HORIZ_OK = 0.4
    DESCEND_ALTITUDE_OK = 0.6

    # Target altitude above the marker per phase. Each phase's target must
    # be *below* the threshold for the next phase, otherwise the drone
    # settles at equilibrium and never transitions. With Kp/Kd ≈ 0.33 in
    # the z PID the drone settles at its target altitude with no
    # overshoot — no slack.
    PHASE_ALTITUDE = {
        PHASE_SEARCH: 5.5,
        PHASE_APPROACH: 3.0,   # below APPROACH_ALTITUDE_OK (3.5)
        PHASE_DESCEND: 0.4,    # below DESCEND_ALTITUDE_OK (0.6)
        PHASE_LAND: -0.5,      # below the marker so the drone keeps descending
    }

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec: DroneSpec = drone_spec
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)
        # Pre-compute the per-motor hover throttle for safe-fallback recovery.
        self._hover_action = self.attitude_ctrl(
            {"attitude": np.zeros(3), "angular_velocity": np.zeros(3)},
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )

        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary, cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            self._detector = None  # OpenCV < 4.7 fallback path used in perceive()

        # Marker corner ordering used by IPPE_SQUARE: TL, TR, BR, BL in the
        # marker's own local frame, marker plane = z=0.
        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [
                [-h,  h, 0.0],
                [ h,  h, 0.0],
                [ h, -h, 0.0],
                [-h, -h, 0.0],
            ],
            dtype=np.float32,
        )

        # PID gains tuned for stability rather than aggressiveness — leave
        # plenty of room for teams to push performance up. With Kp/Kd
        # ratio fixed, the closed-loop velocity at error e is roughly
        # v_steady ≈ (Kp/Kd) * e — ~0.5 means descend at 0.5 m/s for a
        # 1 m altitude error, gentle enough to avoid free-fall before
        # the controller stabilizes.
        self.pid_x = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_y = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_z = PID(kp=0.10, ki=0.02, kd=0.30)
        self.pid_yaw = PID(kp=1.0, ki=0.0, kd=0.0)

        self.phase = PHASE_SEARCH
        self._prev_phase: Optional[str] = None
        self._last_marker_world: Optional[np.ndarray] = None
        self._last_marker_z: Optional[float] = None
        self._frames_since_detection = 0
        self._last_estimate: Optional[Dict] = None

        # Kalman state. Both filters self-initialise on first valid input.
        self._kf = _BoatKalmanFilter(sigma_a=2.0)
        self._heading_kf = _HeadingKalmanFilter()
        self._kf_trusted = False
        self._kf_consecutive_good = 0
        self._kf_consecutive_rejects = 0

    # ------------------------------------------------------------------ public API
    def act(self, obs: Dict) -> np.ndarray:
        """Top-level orchestrator. The agent_template.py version is a
        copy of this; teams should override perceive/estimate/decide/
        control independently and leave act() alone."""
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state)
        self._last_estimate = boat_estimate
        self.phase = self.decide(state, boat_estimate, battery, time_s)
        action = self.control(state, boat_estimate, self.phase)
        if not np.all(np.isfinite(action)):
            import sys
            print(
                f"[BaselineAgent] non-finite action {action}; perception="
                f"{perception}; estimate={boat_estimate}; pos={state['position']}; "
                f"vel={state['velocity']}; rpy={state['attitude']}",
                file=sys.stderr,
            )
            # Safe fallback: hover throttle on every motor.
            action = self._hover_action.copy()
        return action

    def get_last_estimate(self) -> Optional[Dict]:
        """Expose the most recent boat-position estimate so the scorer can
        compute the optional estimation-bonus RMSE. Returning None opts
        out of the bonus."""
        if self._last_estimate is None or self._last_estimate.get("position") is None:
            return None
        return {
            "position": np.asarray(self._last_estimate["position"]),
            "velocity": np.asarray(self._last_estimate.get("velocity", np.zeros(3))),
        }

    # ------------------------------------------------------------------ perception
    def perceive(self, camera_image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)

        if ids is None or len(ids) == 0:
            return {"detected": False}

        # Only marker ID 0 matters for this challenge.
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
        # IPPE_SQUARE silently emits NaN tvec when the four corners are
        # nearly collinear (near the FOV edge, extreme angles). Also
        # reject solutions where the marker is behind the camera or
        # absurdly far. Both manifest as garbage downstream.
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

    # ------------------------------------------------------------------ estimation
    def estimate(self, perception: Dict, drone_state: Dict) -> Dict:
        """4-state CV Kalman on [x, y, vx, vy] + scalar heading filter on
        [psi, psi_dot]. Predict every step; update only on detections that
        pass NIS gating. Predict-only during dropouts (the recovery-bonus
        path). See teams/stick-the-landing/notes/estimate.md for tuning.

        Conforms to teams/stick-the-landing/notes/contracts.md: no history
        arg, dict output with the documented optional fields, never
        propagates non-finite values (emits None per NaN-safety rule).
        """
        dt = self.DT

        # 1) Predict always. The methods no-op until each filter is seeded.
        self._kf.predict(dt)
        self._heading_kf.predict(dt)

        # 2) Validate inputs. NaN/missing drone state => predict-only, no
        # update — that's the safe path the scorer's NaN sanitiser wants.
        drone_pos = np.asarray(drone_state.get("position"), dtype=np.float64)
        attitude = np.asarray(drone_state.get("attitude"), dtype=np.float64)
        if (
            drone_pos.shape != (3,)
            or attitude.shape != (3,)
            or not np.all(np.isfinite(drone_pos))
            or not np.all(np.isfinite(attitude))
        ):
            self._frames_since_detection += 1
            return self._build_estimate_output(updated=False, from_prior=False)

        detected = bool(perception.get("detected", False))
        tvec = perception.get("tvec")
        rvec = perception.get("rvec")
        if detected:
            if tvec is None:
                detected = False
            else:
                tvec = np.asarray(tvec, dtype=np.float64).reshape(-1)
                if tvec.shape != (3,) or not np.all(np.isfinite(tvec)):
                    detected = False
                    tvec = None
        if detected and rvec is not None:
            rvec = np.asarray(rvec, dtype=np.float64).reshape(-1)
            if rvec.shape != (3,) or not np.all(np.isfinite(rvec)):
                rvec = None  # keep the position update, drop heading only

        # 3) Camera-frame -> world transform (same derivation as baseline).
        z_xy: Optional[np.ndarray] = None
        sigma_xy: Optional[float] = None
        psi_meas: Optional[float] = None
        R_cam_to_world: Optional[np.ndarray] = None
        marker_world: Optional[np.ndarray] = None
        if detected:
            R_world_body = rpy_to_matrix(attitude)
            body_x = R_world_body[:, 0]
            body_y = R_world_body[:, 1]
            body_z = R_world_body[:, 2]
            camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec
            if not np.all(np.isfinite(marker_world)):
                detected = False
                marker_world = None
            else:
                self._last_marker_world = marker_world
                self._last_marker_z = float(marker_world[2])
                z_xy = marker_world[:2].astype(np.float64, copy=True)

                # Measurement noise R scales with viewing range (PnP error
                # grows ~linearly with depth) and, when perceive() ships
                # its upgrade, with reprojection error. percept.ambiguous
                # inflates R by 9x (3x sigma).
                alt = max(float(drone_pos[2] - marker_world[2]), 0.5)
                sigma_xy = 0.05 * (alt / 5.0)
                reproj = perception.get("reprojection_error")
                if reproj is not None:
                    reproj_f = float(reproj)
                    if np.isfinite(reproj_f) and reproj_f > 1.0:
                        sigma_xy *= reproj_f
                if bool(perception.get("ambiguous", False)):
                    sigma_xy *= 3.0

                # Heading measurement from rvec, mod pi. The first column
                # of R_marker_world is the marker's local +x axis expressed
                # in world coords; atan2 of its (y, x) components gives psi.
                if rvec is not None:
                    try:
                        R_marker_cam, _ = cv2.Rodrigues(rvec)
                        R_marker_world = R_cam_to_world @ R_marker_cam
                        psi_raw = float(
                            np.arctan2(R_marker_world[1, 0], R_marker_world[0, 0])
                        )
                        if np.isfinite(psi_raw):
                            psi_meas = _wrap_mod_pi(psi_raw)
                    except Exception:
                        psi_meas = None

        # 4) Position-KF measurement update with NIS gating.
        updated_position = False
        if z_xy is not None and sigma_xy is not None:
            R_meas = np.diag([sigma_xy ** 2, sigma_xy ** 2]).astype(np.float64)
            if not self._kf.initialized:
                # First detection: seed the filter directly — no gate.
                self._kf.initialize(z_xy, sigma_xy)
                updated_position = True
                self._kf_consecutive_good = 1
                self._kf_consecutive_rejects = 0
            else:
                innov, S = self._kf.innovation(z_xy, R_meas)
                try:
                    nis = float(innov @ np.linalg.inv(S) @ innov)
                except np.linalg.LinAlgError:
                    nis = float("inf")
                if np.isfinite(nis) and nis < self.NIS_THRESHOLD:
                    self._kf.update(z_xy, R_meas)
                    updated_position = True
                    self._kf_consecutive_good += 1
                    self._kf_consecutive_rejects = 0
                else:
                    self._kf_consecutive_rejects += 1
                    if self._kf_consecutive_rejects >= self.NIS_REINIT_AFTER:
                        # Filter has diverged from reality. Reinit from the
                        # latest measurement rather than rejecting forever.
                        self._kf.initialize(z_xy, sigma_xy)
                        updated_position = True
                        self._kf_consecutive_good = 1
                        self._kf_consecutive_rejects = 0

        # 5) Heading-KF update. Reset-on-flip first (drop the measurement
        # entirely when IPPE_SQUARE has flipped to the mirror pose), then
        # gate on tilt + (when available) confidence.
        if psi_meas is not None and self._heading_kf.initialized:
            if bool(perception.get("ambiguous", False)):
                diff = _wrap_mod_pi(psi_meas - float(self._heading_kf.x[0]))
                if abs(diff) > self.HEADING_FLIP_REJECT_RAD:
                    psi_meas = None

        if psi_meas is not None:
            # Drone tilt = angle between body z and world z.
            cos_tilt = float(np.cos(attitude[0]) * np.cos(attitude[1]))
            tilt = float(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
            confidence_in = perception.get("confidence")
            gate = tilt > self.HEADING_TILT_GATE_RAD
            if confidence_in is not None:
                conf_f = float(confidence_in)
                gate = gate and np.isfinite(conf_f) and (conf_f > self.HEADING_CONFIDENCE_GATE)
            if gate:
                if not self._heading_kf.initialized:
                    self._heading_kf.initialize(psi_meas)
                else:
                    self._heading_kf.update(psi_meas)

        # 6) Trust + stale-step accounting.
        if updated_position:
            self._frames_since_detection = 0
            if self._kf_consecutive_good >= self.KF_TRUST_AFTER:
                self._kf_trusted = True
        else:
            self._frames_since_detection += 1

        # 7) Long-stale handling: gentle covariance inflation past 1.5 s,
        # full divergence reset past 5 s (zero velocity, large cov, prior).
        stale_secs = self._frames_since_detection * dt
        from_prior_now = False
        if stale_secs > self.STALE_INFLATE_AFTER_S and not updated_position:
            excess = stale_secs - self.STALE_INFLATE_AFTER_S
            self._kf.P = self._kf.P * (1.0 + excess)
        if stale_secs > self.STALE_RESET_AFTER_S and self._kf.initialized:
            anchor = (
                self._last_marker_world[:2]
                if self._last_marker_world is not None
                else self._kf.x[:2]
            )
            self._kf.x = np.array(
                [float(anchor[0]), float(anchor[1]), 0.0, 0.0],
                dtype=np.float64,
            )
            self._kf.P = np.diag([4.0, 4.0, 1.0, 1.0]).astype(np.float64)
            self._kf_trusted = False
            self._kf_consecutive_good = 0
            from_prior_now = True

        return self._build_estimate_output(
            updated=updated_position, from_prior=from_prior_now
        )

    def _build_estimate_output(
        self, updated: bool, from_prior: bool = False,
    ) -> Dict:
        """Construct the dict that conforms to contracts.md's
        ``boat_estimate`` schema. NaN-safe: any non-finite scalar/array is
        emitted as ``None`` rather than propagated."""
        # Cold start, no detection ever — same hint as the baseline: a
        # real solution should *search* rather than guess.
        if not self._kf.initialized:
            return {
                "position": np.array(
                    [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3],
                    dtype=np.float64,
                ),
                "velocity": np.zeros(3, dtype=np.float64),
                "fresh": False,
                "stale_steps": int(self._frames_since_detection),
                "from_prior": True,
                "heading": None,
                "heading_rate": None,
                "position_cov": None,
                "velocity_cov": None,
                "confidence": None,
            }

        z_out = self._last_marker_z if self._last_marker_z is not None else 0.3
        pos3 = np.array(
            [float(self._kf.x[0]), float(self._kf.x[1]), float(z_out)],
            dtype=np.float64,
        )
        if self._kf_trusted:
            vel3 = np.array(
                [float(self._kf.x[2]), float(self._kf.x[3]), 0.0],
                dtype=np.float64,
            )
        else:
            vel3 = np.zeros(3, dtype=np.float64)

        # 3x3 covariance views. The z (and vz) entries aren't modelled —
        # we pad with a large finite value so readers can take .trace()
        # without special-casing.
        P_pos3 = np.array(
            [[self._kf.P[0, 0], self._kf.P[0, 1], 0.0],
             [self._kf.P[1, 0], self._kf.P[1, 1], 0.0],
             [0.0,                0.0,                100.0]],
            dtype=np.float64,
        )
        P_vel3 = np.array(
            [[self._kf.P[2, 2], self._kf.P[2, 3], 0.0],
             [self._kf.P[3, 2], self._kf.P[3, 3], 0.0],
             [0.0,                0.0,                100.0]],
            dtype=np.float64,
        )

        trace_pos = float(self._kf.P[0, 0] + self._kf.P[1, 1])
        conf_raw = 1.0 / (1.0 + max(trace_pos, 0.0))
        confidence: Optional[float] = (
            float(conf_raw) if np.isfinite(conf_raw) else None
        )

        if self._heading_kf.initialized:
            h = float(self._heading_kf.x[0])
            hr = float(self._heading_kf.x[1])
            heading: Optional[float] = h if np.isfinite(h) else None
            heading_rate: Optional[float] = hr if np.isfinite(hr) else None
        else:
            heading = None
            heading_rate = None

        # NaN-safety: emit None rather than propagate non-finite outputs.
        if not np.all(np.isfinite(pos3)):
            pos3 = None
            vel3 = None
        elif not np.all(np.isfinite(vel3)):
            vel3 = np.zeros(3, dtype=np.float64)
        if not np.all(np.isfinite(P_pos3)):
            P_pos3 = None
        if not np.all(np.isfinite(P_vel3)):
            P_vel3 = None

        return {
            "position": pos3,
            "velocity": vel3,
            "fresh": bool(updated and self._kf_trusted),
            "stale_steps": int(self._frames_since_detection),
            "from_prior": bool(from_prior),
            "heading": heading,
            "heading_rate": heading_rate,
            "position_cov": P_pos3,
            "velocity_cov": P_vel3,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ decision
    def decide(
        self, drone_state: Dict, boat_estimate: Dict, battery: float, time_s: float
    ) -> str:
        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None or boat_estimate.get("from_prior", False):
            return PHASE_SEARCH

        horiz = float(np.linalg.norm(pos[:2] - np.asarray(target[:2])))
        z_above = float(pos[2] - target[2])

        if z_above < self.DESCEND_ALTITUDE_OK:
            return PHASE_LAND
        if horiz < self.DESCEND_HORIZ_OK and z_above < self.APPROACH_ALTITUDE_OK:
            return PHASE_DESCEND
        if horiz < self.APPROACH_HORIZ_OK:
            return PHASE_APPROACH
        return PHASE_APPROACH if boat_estimate.get("fresh", False) else PHASE_SEARCH

    # ------------------------------------------------------------------ control
    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase: str
    ) -> np.ndarray:
        # Reset PID integrators on phase transitions. Without this, the
        # z-PID's integrator saturates during a prolonged LAND (err is
        # always large-negative against the -0.5 m target) and carries
        # that bias into a subsequent APPROACH, e.g. when a recovery
        # scenario briefly loses the marker and forces re-search.
        if self._prev_phase is not None and phase != self._prev_phase:
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
        self._prev_phase = phase

        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None:
            target = np.array(
                [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]
            )

        target_z = float(target[2]) + self.PHASE_ALTITUDE[phase]

        # World-frame errors -> body-frame errors so the controller works
        # regardless of yaw. The baseline never actively yaws, but this
        # keeps things correct if anything perturbs heading.
        R = rpy_to_matrix(drone_state["attitude"])
        vel_world = np.asarray(drone_state["velocity"], dtype=np.float64)
        vel_body = R.T @ vel_world
        err_world = np.array(
            [target[0] - pos[0], target[1] - pos[1], target_z - pos[2]],
            dtype=np.float64,
        )
        err_body = R.T @ err_world

        # Sign mapping: a positive forward (body +x) error means the target
        # is ahead, which calls for a positive pitch (which rotates body z
        # forward, accelerating +x). A positive left (body +y) error means
        # target is to the left, which calls for a negative roll
        # (-roll_cmd accelerates +y).
        #
        # Thrust uses err_world[2] (NOT err_body[2]): altitude is
        # controlled in the world frame, not the body frame.
        #
        # PID D-term uses drone velocity ('D on measurement'), which is
        # smooth, instead of d(err)/dt, which spikes when the noisy boat
        # estimate snaps to a new detection. With d(err)/dt the
        # controller bang-bangs ±1 every step and the simulation
        # numerically diverges.
        pitch_cmd = self.pid_x(err_body[0], self.DT, measurement_velocity=vel_body[0])
        roll_cmd = -self.pid_y(err_body[1], self.DT, measurement_velocity=vel_body[1])
        thrust_cmd = self.pid_z(err_world[2], self.DT, measurement_velocity=vel_world[2])
        # In LAND, force thrust to the controller's lower bound (50% hover
        # thrust). The z-PID alone outputs ~ -0.10 at this altitude — over
        # the platform the reference sim's ground-effect amplification cancels
        # that small negative bias and the drone hovers ~0.2 m above the deck
        # forever. Full negative thrust pushes through the GE cushion (cap is
        # 2× per-motor in the ref sim, so worst-case F_eff ≈ mg at h ≈ 0.05 m)
        # and the drone touches down. Crude — descent is uncontrolled and
        # max_descent_velocity often saturates, killing the soft-landing
        # bonus. Smooth descent profiles + velocity feed-forward are left
        # as exercises.
        if phase == PHASE_LAND:
            thrust_cmd = -1.0
        yaw_rate_cmd = 0.0

        # Hand the high-level (thrust, roll, pitch, yaw_rate) setpoints to
        # the stock attitude controller, which mixes them into per-motor
        # throttles using the drone spec's geometry and propeller params.
        return self.attitude_ctrl(
            drone_state,
            roll_des=roll_cmd,
            pitch_des=pitch_cmd,
            yaw_rate_des=yaw_rate_cmd,
            thrust_norm=thrust_cmd,
        )


# Module-level entry point. evaluation/evaluate.py imports the module
# dynamically and looks for a callable named `make_agent` (preferred) or a
# class named `Agent`. We support both for convenience.
Agent = BaselineAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> BaselineAgent:
    return BaselineAgent(drone_spec)
