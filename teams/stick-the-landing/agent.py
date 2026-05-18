"""StickTheLanding — competition agent for the Catch the Boat challenge.

Forked from agents/agent_template.py (the documented starting point), so
`act()` is the locked orchestrator and the four pipeline stages are filled
in independently:

    perceive()  — Person 1: ArUco detection + pose recovery  [DONE]
    estimate()  — Person 2: Kalman filter on boat (x, y, vx, vy) + heading
    decide()    — Person 3: SEARCH / APPROACH / DESCEND / LAND state machine
    control()   — Person 3: active yaw alignment + smooth descent

The current perception deliverable is the `perceive()` method plus the
module-level `marker_to_world()` helper; estimate()/decide()/control() are
still the template's stubs, so the agent does not land until they are filled.

Each stage method keeps the template's improvement-idea docstring as a guide
for Persons 2 & 3.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


# agent.py lives in teams/<team>/, so the repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"


def rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation matrix, PyBullet RPY convention
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def marker_to_world(perception: Dict, drone_state: Dict):
    """Transform a camera-frame detection into a world-frame marker pose.

    Returns (marker_world (3,), marker_yaw) or (None, None) when `perception`
    holds no detection. The OpenCV camera axes map to the drone body axes as
        cam_X = -body_y,  cam_Y = -body_x,  cam_Z = -body_z
    (derived from the env's downward-camera view-matrix construction). Person 2's
    estimate() calls this, then runs the Kalman filter on `marker_world`.
    """
    if not perception.get("detected", False):
        return None, None

    tvec = np.asarray(perception["tvec"], dtype=np.float64).reshape(3)
    pos = np.asarray(drone_state["position"], dtype=np.float64).reshape(3)
    R_world_body = rpy_to_matrix(drone_state["attitude"])
    body_x, body_y, body_z = (
        R_world_body[:, 0], R_world_body[:, 1], R_world_body[:, 2]
    )
    # The camera sits CAMERA_BODY_OFFSET_Z along body z, below the COM.
    camera_world = pos + body_z * CAMERA_BODY_OFFSET_Z
    R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
    marker_world = camera_world + R_cam_to_world @ tvec

    # Boat heading from the marker's in-plane rotation. The marker is painted
    # axis-aligned on the platform, so its x-axis tracks the boat length axis
    # up to a fixed offset the controller calibrates (landing is yaw mod pi).
    R_mc, _ = cv2.Rodrigues(np.asarray(perception["rvec"], dtype=np.float64))
    R_mw = R_cam_to_world @ R_mc
    marker_yaw = float(np.arctan2(R_mw[1, 0], R_mw[0, 0]))
    return marker_world, marker_yaw


class TemplateAgent:
    DT = 0.02  # must match BoatLandingEnv.DT

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        # The env passes the active drone spec at construction. If you
        # instantiate this agent standalone (tests, notebooks), it falls
        # back to drones/vtol.yaml.
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec: DroneSpec = drone_spec
        # Default attitude controller: maps (thrust, roll, pitch, yaw_rate)
        # setpoints to per-motor throttles. Replace with your own if you
        # want full motor-level control.
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)

        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # --- Perception state (Person 1) ---
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # Detector tuning: subpixel corner refinement (a real solvePnP gain)
        # and a widened adaptive-threshold window range, which helps the
        # detector survive the HARD scenario's fog and horizontal motion blur.
        self._params = cv2.aruco.DetectorParameters()
        self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._params.adaptiveThreshWinSizeMin = 3
        self._params.adaptiveThreshWinSizeMax = 43
        self._params.adaptiveThreshWinSizeStep = 10
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)
        except AttributeError:
            self._detector = None  # OpenCV < 4.7 fallback path used in perceive()

        # CLAHE: local contrast equalisation, for fog robustness.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Last raw-frame hash, for stale-frame detection in perceive().
        self._last_frame_hash: Optional[int] = None
        # Marker corner points in the marker's own frame, TL/TR/BR/BL — the
        # order cv2.aruco.detectMarkers returns, as IPPE_SQUARE expects.
        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [[-h,  h, 0.0], [ h,  h, 0.0], [ h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )

        # --- Estimation / control state (Persons 2 & 3 fill these in) ---
        self.phase = PHASE_SEARCH
        self._last_estimate: Optional[Dict] = None

    # ------------------------------------------------------------------ DO NOT MODIFY
    def act(self, obs: Dict) -> np.ndarray:
        """Top-level orchestrator. Identical to the baseline's. Leave
        this alone so the four pipeline stages stay easy to compare
        across agents."""
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state, history=None)
        self._last_estimate = boat_estimate
        self.phase = self.decide(state, boat_estimate, battery, time_s)
        action = self.control(state, boat_estimate, self.phase)
        return action

    def get_last_estimate(self) -> Optional[Dict]:
        """Optional: return your latest boat estimate for the scorer's
        estimation bonus. Should return a dict with keys 'position'
        (np.ndarray of shape (3,)) and optionally 'velocity'.

        Return None to opt out of the bonus."""
        return self._last_estimate

    # ------------------------------------------------------------------ OPTIONAL: FC-compatible output
    # If you implement `act_setpoint` (instead of, or in addition to,
    # `act`), the evaluator calls it INSTEAD of `act`. It must return
    # `(thrust_norm, roll, pitch, yaw_rate)` — the same 4-DoF setpoint
    # a real flight controller accepts via MAVLink SET_ATTITUDE_TARGET.
    # The runner converts those four numbers to per-motor throttles via
    # `DefaultAttitudeController` before calling env.step, so you do NOT
    # need to run the attitude controller yourself.
    #
    # Why bother:
    #   - Agents that emit setpoints port to PX4/Ardupilot OFFBOARD in an
    #     afternoon. Agents that emit motor throttles need a custom FC.
    #   - +5 HW-readiness points (see docs/AGENT_SCORING.md).
    #
    # Example (uncomment + adapt):
    #
    # def act_setpoint(self, obs: Dict) -> tuple:
    #     camera = obs["camera"]; state = obs["state"]
    #     perception   = self.perceive(camera)
    #     boat_est     = self.estimate(perception, state, history=None)
    #     self._last_estimate = boat_est
    #     self.phase   = self.decide(state, boat_est, obs["battery"], obs["time"])
    #     thrust_norm, roll, pitch, yaw_rate = self.control_setpoint(
    #         state, boat_est, self.phase
    #     )
    #     return (thrust_norm, roll, pitch, yaw_rate)

    # ------------------------------------------------------------------ STAGE 1: PERCEPTION
    def perceive(self, camera_image: np.ndarray) -> Dict:
        """Detect marker 0 in one RGB frame and recover its camera-frame pose.

        Returns a dict: always {"detected", "stale"}; when detected also
        {"tvec", "rvec", "image_points", "reproj_error", "marker_px"}.
        `marker_to_world()` (module level) turns this into a world-frame pose.

        Perception never smooths or predicts — it reports each frame honestly,
        including "no detection". Filtering is Person 2's estimate() stage.
        """
        # Stale-frame detection: when the scenario fps < 50 the env repeats
        # the previous frame. Hashing the raw bytes lets estimate() skip a
        # duplicate measurement (applying it twice falsely shrinks the KF
        # covariance).
        frame_hash = hash(camera_image.tobytes())
        stale = frame_hash == self._last_frame_hash
        self._last_frame_hash = frame_hash

        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        # CLAHE restores marker contrast under the Beer-Lambert fog blend.
        gray = self._clahe.apply(gray)

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )

        if ids is None or len(ids) == 0:
            return {"detected": False, "stale": stale}

        # Only marker ID 0 matters for this challenge.
        target = None
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                target = c
                break
        if target is None:
            return {"detected": False, "stale": stale}

        image_points = np.asarray(target, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            self.intrinsics,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return {"detected": False, "stale": stale}

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        # IPPE_SQUARE silently emits NaN tvec when the four corners are
        # nearly collinear (near the FOV edge, extreme angles). Also reject
        # solutions where the marker is behind the camera or absurdly far.
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False, "stale": stale}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False, "stale": stale}

        # Reprojection error (RMS corner distance, px) — a quality metric the
        # estimator turns into a Kalman measurement-noise weight.
        proj, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, self.intrinsics, self.dist_coeffs
        )
        proj = proj.reshape(4, 2)
        reproj_error = float(
            np.sqrt(np.mean(np.sum((proj - image_points) ** 2, axis=1)))
        )
        # Apparent marker size in px — a distance / confidence proxy.
        marker_px = float(
            np.mean([
                np.linalg.norm(image_points[i] - image_points[(i + 1) % 4])
                for i in range(4)
            ])
        )

        return {
            "detected": True,
            "stale": stale,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
            "reproj_error": reproj_error,
            "marker_px": marker_px,
        }

    # ------------------------------------------------------------------ STAGE 2: ESTIMATION
    def estimate(
        self, perception: Dict, drone_state: Dict, history
    ) -> Dict:
        """Fuse the perception result with the drone state (and optionally
        a history buffer) to produce a boat estimate.

        Must return at minimum:
            {'position': np.ndarray shape (3,) | None,
             'velocity': np.ndarray shape (3,)}

        BASELINE: trusts every detection blindly; for missing detections
        it returns the last known position; with no history at all it
        returns a hardcoded prior (origin). No filtering, no velocity.

        WHY IT'S WEAK:
        - No measurement model: a noisy frame can produce a noisy world
          estimate that the controller will then chase.
        - No motion model: the boat is moving but we estimate velocity
          as zero, so the controller always lags.
        - Lost detections trigger no recovery behavior.

        IMPROVEMENT IDEAS:
        1. KALMAN FILTER on (x, y, vx, vy). State updates with constant-
           velocity model, measurements come from solvePnP. This single
           change typically dwarfs every other improvement.
        2. Use the drone-state velocity to subtract ego-motion when
           checking detection consistency.
        3. Outlier rejection: discard detections whose reprojection error
           or jump-from-previous is too large.
        4. Estimate boat heading and yaw rate too (from successive
           position estimates or from the marker's rvec).
        5. When detection is lost, propagate the filter forward (predict
           only) for a few frames before reverting to search.
        """
        # TODO: replace with your estimator.
        return {"position": None, "velocity": np.zeros(3)}

    # ------------------------------------------------------------------ STAGE 3: DECISION
    def decide(
        self,
        drone_state: Dict,
        boat_estimate: Dict,
        battery: float,
        time_s: float,
    ) -> str:
        """Pick a high-level phase based on the current state. Must
        return one of:
            PHASE_SEARCH, PHASE_APPROACH, PHASE_DESCEND, PHASE_LAND

        BASELINE: thresholds on horizontal distance and altitude above
        the marker, with a tiny hysteresis (returns SEARCH if the
        estimate is from the cold-start prior).

        WHY IT'S WEAK:
        - No timing awareness: descends regardless of whether the boat
          is at the worst moment of its roll oscillation.
        - No battery-aware decisions: doesn't abort if energy is low.
        - No 'go-around' behavior: once committed to LAND, can't bail
          and re-approach if drift takes it off the platform.

        IMPROVEMENT IDEAS:
        1. Wait until the boat's roll/pitch oscillation crosses zero
           before committing to LAND (estimate the phase from your
           filter or by tracking the marker's rvec over time).
        2. Add an ABORT phase: if descent is going wrong (e.g. drone is
           drifting off-platform, or the estimate becomes stale), climb
           and re-approach.
        3. Speed-vs-battery trade-off: if battery is low, accept a less
           precise landing rather than burning more time hovering.
        4. Time-budgeting: estimate remaining time-to-land and bail to
           a faster (less precise) path if you'd otherwise time out.
        """
        # TODO: replace with your decision policy.
        return PHASE_SEARCH

    # ------------------------------------------------------------------ STAGE 4: CONTROL
    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase: str
    ) -> np.ndarray:
        """Produce the per-motor throttle action of shape
        `(self.spec.num_motors,)` in `[0, 1]`.

        Two valid pipelines:

        (A) Use the stock attitude controller (matches the baseline):

                roll_des = ...      # in [-1, +1] -> mapped to ±max_tilt
                pitch_des = ...
                yaw_rate_des = ...
                thrust_norm = ...   # in [-1, +1]; 0 = hover
                return self.attitude_ctrl(
                    drone_state, roll_des, pitch_des, yaw_rate_des, thrust_norm
                )

        (B) Emit motor throttles directly (more flexibility, possible
            sim-quality bonus):

                throttles = my_motor_mixing(...)
                return np.clip(throttles, 0.0, 1.0)

        BASELINE: three independent PIDs (x, y, z) on body-frame errors,
        feeding (A) above with the result. No yaw control, constant per-
        phase target altitudes, no feed-forward. In LAND phase the thrust
        PID is overridden with thrust_norm=-1 (bang-bang minimum) so the
        drone punches through the reference sim's ground-effect cushion
        — that's how the baseline lands EASY at all, but the descent is
        crude and forfeits the soft-landing bonus.

        WHY IT'S WEAK:
        - Independent axes ignore the cross-coupling caused by tilt.
        - No feed-forward on the boat's velocity, so we always chase.
        - No wind compensation.
        - LAND descent is bang-bang — fast but rough, no soft-landing
          bonus on touchdown.
        - No active yaw to align fuselage with boat heading — the
          wing-perpendicular landing condition will fail unless the boat
          happens to be aligned with world x at touchdown.

        IMPROVEMENT IDEAS:
        1. ACTIVE YAW CONTROL: command yaw_rate to drive your estimate
           of (boat_heading - drone_yaw) toward 0 or pi. On the VTOL the
           yaw response is slow — start aligning during APPROACH.
        2. FEED-FORWARD: command a target velocity equal to the boat's
           estimated velocity plus a correction term.
        3. Cascaded structure: outer loop on position -> velocity
           setpoint, inner loop on velocity -> attitude setpoint.
        4. Wind feed-forward: estimate wind from steady-state attitude
           offset and add a cancelation term to thrust direction.
        5. SOFT DESCENT PROFILE: replace baseline's bang-bang LAND with a
           target_z(t) that decreases monotonically and matches a
           smooth |v_z| < 1 m/s on touchdown — recovers the +5 soft
           landing bonus the baseline gives up.
        6. Light MPC: optimize the next 1-2 s over a low-DOF horizon.
        """
        # TODO: replace with your controller. Default = hover throttle.
        return self.attitude_ctrl(
            drone_state,
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )


# evaluation/evaluate.py looks for `make_agent` first, then `Agent`.
Agent = TemplateAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> TemplateAgent:
    return TemplateAgent(drone_spec)
