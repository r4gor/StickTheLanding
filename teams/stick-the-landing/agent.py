
"""StickTheLanding — competition agent for the Catch the Boat challenge.

Forked from agents/agent_template.py (the documented starting point), so
`act()` is the locked orchestrator and the four pipeline stages are filled
in independently:

    perceive()  — Person 1: ArUco detection + pose recovery  [DONE]
    estimate()  — Person 2: Kalman filter on boat (x, y, vx, vy) + heading  [DONE]
    decide()    — Person 3: SEARCH / APPROACH / DESCEND / LAND state machine
    control()   — Person 3: active yaw alignment + smooth descent

perceive() and estimate() are competition-grade; decide()/control() are the
baseline-quality 3-PID + threshold state machine from agent_template.py and
are next on the critical path (see teams/stick-the-landing/notes/control.md).
"""

import os
import time
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


def marker_to_world(perception: Dict, drone_state: Dict):
    """Transform a camera-frame detection into a world-frame marker pose.

    Returns (marker_world (3,), marker_yaw) or (None, None) when `perception`
    holds no detection. The OpenCV camera axes map to the drone body axes as
        cam_X = -body_y,  cam_Y = -body_x,  cam_Z = -body_z
    (derived from the env's downward-camera view-matrix construction).
    estimate() performs the same transform inline so the Kalman filter can
    scale R with altitude; this helper is exposed for callers that just want
    the world-frame pose without the filter.
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
    # up to a fixed offset (landing condition is yaw mod pi).
    R_mc, _ = cv2.Rodrigues(np.asarray(perception["rvec"], dtype=np.float64))
    R_mw = R_cam_to_world @ R_mc
    marker_yaw = float(np.arctan2(R_mw[1, 0], R_mw[0, 0]))
    return marker_world, marker_yaw


def _wrap_mod_pi(angle: float) -> float:
    """Wrap an angle into (-pi/2, pi/2]. The landing condition is fuselage
    alignment mod pi (both 0 and pi pass), so heading lives in a half-revolution
    range — this is the canonical representative."""
    return ((float(angle) + np.pi / 2.0) % np.pi) - np.pi / 2.0


class _BoatKalmanFilter:
    """5-state coordinated-turn EKF on world-frame ``[x, y, vx, vy, omega]``.

    The boat in HARD follows a constant-radius curve (r=30 m at 1.8 m/s,
    omega = v/r ~ 0.06 rad/s). A pure constant-velocity (CV) filter
    absorbs the 0.11 m/s² centripetal accel into Q but is biased during
    sustained turns — fine for the est-bonus RMSE, painful during a
    long blackout (predict-only drifts off the arc). Adding omega to the
    state lets the filter predict the curve forward correctly.

    Dynamics (continuous):
        x_dot   = vx
        y_dot   = vy
        vx_dot  = -vy * omega         # rotate velocity vector
        vy_dot  =  vx * omega
        omega_dot = 0                  # nearly-constant-turn model

    With dt=0.02 s and |omega·dt| ~ 0.0012 rad on HARD, the small-angle
    EKF Jacobian is plenty accurate (the higher-order Taylor terms are
    ~5e-7 per step). When the boat is straight (omega → 0 with Q-pull),
    the EKF reduces to the CV filter to machine precision — preserves
    the test_kf_estimator regression suite.

    ``sigma_a`` is the white-noise-acceleration std on (vx, vy) [m/s^2];
    ``sigma_omegadot`` is the same on omega [rad/s^2]. A boat doesn't
    change turn rate fast — sigma_omegadot=0.05 lets omega track the
    HARD curve in <2 s while staying near 0 on EASY (where the boat is
    static).
    """

    # 2x5 measurement matrix — z = [x_meas, y_meas]; vx, vy, omega are
    # observed only through their effect on position over time.
    H = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    def __init__(
        self, sigma_a: float = 2.0, sigma_omegadot: float = 0.05,
    ) -> None:
        self.x = np.zeros(5, dtype=np.float64)
        self.P = np.eye(5, dtype=np.float64) * 1.0e6  # huge prior — uninitialised
        self.initialized = False
        self.sigma_a = float(sigma_a)
        self.sigma_omegadot = float(sigma_omegadot)

    def _F(self, dt: float) -> np.ndarray:
        """Discrete-time linearised state-transition Jacobian, evaluated at
        the current state. Closed-form Taylor expansion is exact through
        first order in dt; on this problem |omega·dt| << 1 makes the
        truncation negligible."""
        vx = float(self.x[2])
        vy = float(self.x[3])
        omega = float(self.x[4])
        return np.array(
            [[1.0, 0.0, dt,         0.0,        0.0],
             [0.0, 1.0, 0.0,        dt,         0.0],
             [0.0, 0.0, 1.0,        -omega * dt, -vy * dt],
             [0.0, 0.0, omega * dt,  1.0,        vx * dt],
             [0.0, 0.0, 0.0,        0.0,        1.0]],
            dtype=np.float64,
        )

    def _Q(self, dt: float) -> np.ndarray:
        """White-noise process noise: WNA on (vx, vy) (preserves the CV
        block 1:1 when omega=0) plus an independent WNA on omega.
        Cross-terms are zero — the angular and linear accelerations of
        the boat are physically uncorrelated to first order."""
        s = self.sigma_a ** 2
        so = self.sigma_omegadot ** 2
        Q = np.zeros((5, 5), dtype=np.float64)
        Q[0, 0] = s * dt**4 / 4.0
        Q[1, 1] = s * dt**4 / 4.0
        Q[0, 2] = s * dt**3 / 2.0
        Q[2, 0] = s * dt**3 / 2.0
        Q[1, 3] = s * dt**3 / 2.0
        Q[3, 1] = s * dt**3 / 2.0
        Q[2, 2] = s * dt**2
        Q[3, 3] = s * dt**2
        Q[4, 4] = so * dt
        return Q

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        # Nonlinear state propagation (forward Euler is sufficient at
        # dt=0.02; the EKF's first-order Jacobian assumes Euler too).
        x_prev, y_prev = self.x[0], self.x[1]
        vx_prev, vy_prev = self.x[2], self.x[3]
        omega = self.x[4]
        self.x = np.array(
            [x_prev + vx_prev * dt,
             y_prev + vy_prev * dt,
             vx_prev - vy_prev * omega * dt,
             vy_prev + vx_prev * omega * dt,
             omega],
            dtype=np.float64,
        )
        F = self._F(dt)
        self.P = F @ self.P @ F.T + self._Q(dt)

    def innovation(self, z: np.ndarray, R: np.ndarray):
        """Return ``(innov, S)`` for outlier gating before committing."""
        innov = np.asarray(z, dtype=np.float64) - self.H @ self.x
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
        self.P = (np.eye(5) - K @ self.H) @ self.P

    def initialize(self, z: np.ndarray, sigma_xy: float) -> None:
        # Seed omega=0 with a moderately confident prior (sigma_omega
        # ~ 0.1 rad/s). Too confident and a HARD curve takes seconds to
        # detect; too loose and EASY's static boat picks up phantom turn
        # rate from measurement noise. 0.1 rad/s splits the difference.
        sigma_omega_init = 0.1
        self.x = np.array(
            [float(z[0]), float(z[1]), 0.0, 0.0, 0.0], dtype=np.float64,
        )
        self.P = np.diag(
            [sigma_xy ** 2, sigma_xy ** 2, 1.0, 1.0, sigma_omega_init ** 2]
        ).astype(np.float64)
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
        # D-term first so the anti-windup decision below sees the full
        # pre-integration command. measurement_velocity (D-on-measurement)
        # is preferred — d(err)/dt spikes when the noisy boat estimate
        # snaps to a new detection.
        if measurement_velocity is not None:
            d = -float(measurement_velocity)
        elif self._prev_err is None:
            d = 0.0
        else:
            d = (err - self._prev_err) / max(dt, 1e-6)
        self._prev_err = err

        # Conditional-integration anti-windup (control.md item #7): refuse
        # to integrate further into saturation when the unclipped command
        # is already pushed past the actuator limit IN THE SAME DIRECTION
        # as the current error. Standard clamp-on-saturate form.
        out_unclipped = self.kp * err + self.ki * self._i + self.kd * d
        out = float(np.clip(out_unclipped, self.out_min, self.out_max))
        sat_limit = max(abs(self.out_max), abs(self.out_min))
        saturated_against_err = (
            abs(out_unclipped) > sat_limit
            and (out_unclipped >= 0.0) == (err >= 0.0)
        )
        if not saturated_against_err:
            self._i = float(np.clip(self._i + err * dt, -self.i_clip, self.i_clip))
        return out


class BaselineAgent:
    """ArUco + state machine + 3 independent PIDs."""

    DT = 0.02  # must match BoatLandingEnv.DT

    # Position-KF outlier gating. 9.0 ≈ chi^2 99% with 2 DoF — Mahalanobis
    # distance ≈ 3 in 2D-equivalent terms ("3σ rejection"). Tightened from
    # the previous 13.8 (99.9%) because on HARD a single fog/blur frame
    # can produce a catastrophically bad PnP solution; we'd rather drop
    # the detection than yank a converged track sideways. Five consecutive
    # rejects still trip a reinit-from-latest so the filter recovers from
    # sustained measurement shifts (e.g. after a long blackout).
    NIS_THRESHOLD = 9.0
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

        # --- Perception state (Person 1) ---
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # Detector tuning: AprilTag saddle-point corner refinement (halves tvec
        # jitter vs SUBPIX) with a SUBPIX fallback for older OpenCV builds,
        # widened adaptive-threshold range + relaxed polygon constraints to
        # survive the HARD scenario's fog, motion blur, and small markers at
        # altitude.
        self._params = cv2.aruco.DetectorParameters()
        try:
            self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        except AttributeError:
            self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._params.adaptiveThreshWinSizeMin = 3
        self._params.adaptiveThreshWinSizeMax = 53
        self._params.adaptiveThreshWinSizeStep = 10
        self._params.polygonalApproxAccuracyRate = 0.05
        self._params.minMarkerPerimeterRate = 0.02
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

        # Dark-channel defog parameters (He et al. 2009). Always-on, runs at
        # 1/4 resolution and the transmission map is upsampled to full size.
        self._defog_omega = 0.95
        self._defog_t0 = 0.1
        self._defog_scale = 4
        self._defog_gf_radius = 15
        self._defog_gf_eps = 1e-3
        self._defog_kernel = np.ones((7, 7), dtype=np.uint8)
        # Fog gate: dark-channel defog is the dominant perceive() cost on
        # clean frames (EASY scenario: 0 fog → 6-10 ms wasted per call).
        # A truly-foggy frame raises the minimum pixel value across all
        # channels (Beer-Lambert blends towards atmospheric light, so dark
        # pixels lose brightness contrast). If a 1/64-subsample finds any
        # genuinely dark pixel, skip defog entirely — it'd be a near-identity
        # transform anyway since dark_channel ≈ 0 implies transmission ≈ 1.
        self._defog_skip_thresh = 30

        # Stage-timing instrumentation. Off by default (cheap; ~100 ns per
        # time.perf_counter() call, ~2 µs total per perceive()). Enable
        # via AGENT_TIMING=1 to print rolling mean-ms-per-stage every
        # _TIMING_PRINT_EVERY perceive() calls — useful for profiling but
        # noisy enough that we don't want it on the scored run.
        self._timing_enabled = bool(os.environ.get("AGENT_TIMING"))
        self._TIMING_PRINT_EVERY = 50
        self._timing_stage_us: Dict[str, float] = {
            "hash": 0.0, "defog_skip_test": 0.0, "defog_run": 0.0,
            "gray_clahe": 0.0, "aruco": 0.0, "solvepnp_aruco": 0.0,
            "wiener": 0.0, "aruco2": 0.0, "cnn": 0.0, "total": 0.0,
        }
        self._timing_run_counts: Dict[str, int] = {
            "perceive": 0, "defog_run": 0, "wiener": 0, "cnn": 0,
        }

        # Wiener deconvolution kernel for horizontal 1x3 box motion blur,
        # the worst case the env applies (camera.apply_motion_blur).
        # Precomputed FFT — used on the fallback path when ArUco misses.
        # Kernel is centered (zero-phase) so the deconvolution preserves
        # pixel alignment.
        self._wiener_nsr = 0.01
        _kpad = np.zeros((480, 640), dtype=np.float32)
        _kpad[0, -1] = 1.0 / 3.0
        _kpad[0, 0] = 1.0 / 3.0
        _kpad[0, 1] = 1.0 / 3.0
        self._wiener_K = np.fft.rfft2(_kpad)

        # CNN corner-regressor fallback. Lazy-loaded; failure non-fatal.
        self._cnn_net = None
        self._cnn_ready = False
        self._try_load_cnn()

        # Cache of the previous perception dict — stale frames (the env's
        # camera fps < 50 produces byte-identical repeats) return the cached
        # result instead of re-running defog/ArUco/Wiener. estimate() still
        # ignores stale measurements; the cache is purely a latency win.
        self._last_perception: Optional[Dict] = None

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
        phase_out = self.decide(state, boat_estimate, battery, time_s)
        # decide() may eventually return a dict per contracts.md; for now
        # the baseline returns a string. Surface the string form on self
        # for legacy inspectors, but pass the full phase_out into control().
        self.phase = self._phase_from(phase_out)
        action = self.control(state, boat_estimate, phase_out)
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

    # ------------------------------------------------------------------ STAGE 1: PERCEPTION
    # Per-source confidence prior. ArUco direct is trusted; deconv-recovered
    # detections get a modest haircut; CNN-regressed corners are the most
    # uncertain. estimate() turns confidence into R-scale at line ~617.
    _SOURCE_PRIOR = {"aruco": 1.0, "deconv": 0.7, "cnn": 0.5}

    def perceive(self, camera_image: np.ndarray) -> Dict:
        """Detect marker 0 in one RGB frame and recover its camera-frame pose.

        Returns a dict: always {"detected", "stale"}; when detected also
        {"tvec", "rvec", "image_points", "reprojection_error", "confidence",
        "ambiguous", "source"} per teams/stick-the-landing/notes/contracts.md.
        `marker_to_world()` (module level) turns this into a world-frame pose.

        Cascade (HARD/recovery survivability — Tip #5):
            rgb -> dark_channel_defog -> gray -> CLAHE -> ArUco        ("aruco")
                (miss) -> Wiener deblur -> ArUco                       ("deconv")
                (miss) -> CNN corner regressor + solvePnP              ("cnn")
                (miss) -> {detected: False}

        Defog is always-on (cheap; recovers fog-attenuated contrast on top of
        CLAHE). Wiener and CNN run only on miss to avoid ringing/CNN-noise on
        clean frames. The CNN path is also latency-gated.

        Perception never smooths or predicts — it reports each frame honestly,
        including "no detection". Filtering is estimate()'s job.
        """
        t0 = time.perf_counter()
        tlog = self._timing_enabled

        # Stale-frame detection: when the scenario fps < 50 the env repeats
        # the previous frame. Hashing the raw bytes lets estimate() skip a
        # duplicate measurement (applying it twice falsely shrinks the KF
        # covariance).
        frame_hash = hash(camera_image.tobytes())
        stale = frame_hash == self._last_frame_hash
        self._last_frame_hash = frame_hash
        if tlog:
            tA = time.perf_counter()
            self._timing_stage_us["hash"] += (tA - t0) * 1e6

        # Stale short-circuit: re-running defog + ArUco on a byte-identical
        # frame produces the same answer at full cost. Return the cached dict
        # with stale=True so estimate() drops it but downstream tooling still
        # sees the previous detection state.
        if stale and self._last_perception is not None:
            cached = dict(self._last_perception)
            cached["stale"] = True
            if tlog:
                self._record_perceive_timing(t0)
            return cached

        # A: Defog (fog-gated). On a clean frame the gate returns rgb
        # unchanged in ~50 µs; on a foggy frame this is the 3-5 ms path.
        if tlog:
            tB = time.perf_counter()
        rgb_defogged = self._defog_dark_channel(camera_image)
        if tlog:
            tC = time.perf_counter()
            # Split skip-test from real-work: the gate's cost is the elapsed
            # if the output IS the input (id-equal); otherwise it's the full
            # defog path.
            if rgb_defogged is camera_image:
                self._timing_stage_us["defog_skip_test"] += (tC - tB) * 1e6
            else:
                self._timing_stage_us["defog_run"] += (tC - tB) * 1e6
                self._timing_run_counts["defog_run"] += 1

        gray = cv2.cvtColor(rgb_defogged, cv2.COLOR_RGB2GRAY)
        # CLAHE restores marker contrast under the Beer-Lambert fog blend.
        gray = self._clahe.apply(gray)
        if tlog:
            tD = time.perf_counter()
            self._timing_stage_us["gray_clahe"] += (tD - tC) * 1e6

        image_points, n_rejected = self._aruco_detect(gray)
        if tlog:
            tE = time.perf_counter()
            self._timing_stage_us["aruco"] += (tE - tD) * 1e6
        if image_points is not None:
            result = self._solve_pose_from_corners(
                image_points, stale, source="aruco"
            )
            if tlog:
                tF = time.perf_counter()
                self._timing_stage_us["solvepnp_aruco"] += (tF - tE) * 1e6
            if result["detected"]:
                self._last_perception = result
                if tlog:
                    self._record_perceive_timing(t0)
                return result

        # B: Wiener deconv fallback — only when ArUco saw quad candidates
        # but couldn't decode marker 0. If n_rejected == 0 the frame has no
        # marker-shaped polygons; deblur cannot synthesize them. Also
        # latency-gated: on heavy noise the first ArUco pass alone burns
        # the budget; a second pass would push perceive() above 100 ms.
        if n_rejected > 0 and (time.perf_counter() - t0) < 0.012:
            if tlog:
                tG = time.perf_counter()
                self._timing_run_counts["wiener"] += 1
            gray_dbl = self._deblur_wiener(gray)
            result = self._aruco_and_pose(gray_dbl, stale, source="deconv")
            if tlog:
                tH = time.perf_counter()
                self._timing_stage_us["wiener"] += (tH - tG) * 1e6
            if result["detected"]:
                self._last_perception = result
                if tlog:
                    self._record_perceive_timing(t0)
                return result

        # C: CNN corner regressor — latency-gated. Runs even when n_rejected
        # is 0 because the CNN can pick up the marker from low-level features
        # ArUco's polygon-first pipeline misses (e.g. heavy fog + small marker).
        if self._cnn_ready and (time.perf_counter() - t0) < 0.015:
            if tlog:
                tI = time.perf_counter()
                self._timing_run_counts["cnn"] += 1
            pred_corners = self._cnn_corners(camera_image)
            if pred_corners is not None:
                result = self._solve_pose_from_corners(
                    pred_corners, stale, source="cnn"
                )
                if tlog:
                    self._timing_stage_us["cnn"] += (time.perf_counter() - tI) * 1e6
                if result["detected"]:
                    self._last_perception = result
                    if tlog:
                        self._record_perceive_timing(t0)
                    return result
            elif tlog:
                self._timing_stage_us["cnn"] += (time.perf_counter() - tI) * 1e6

        miss = {"detected": False, "stale": stale}
        self._last_perception = miss
        if tlog:
            self._record_perceive_timing(t0)
        return miss

    def _record_perceive_timing(self, t0: float) -> None:
        """Accumulate total perceive() time and emit a TIMING line every
        _TIMING_PRINT_EVERY calls. Mean-µs-per-call is reported per stage,
        with run-count denominators for the optional stages."""
        self._timing_stage_us["total"] += (time.perf_counter() - t0) * 1e6
        self._timing_run_counts["perceive"] += 1
        n = self._timing_run_counts["perceive"]
        if n % self._TIMING_PRINT_EVERY:
            return
        import sys
        # Per-stage average over all perceives (i.e., amortized cost,
        # including the 0-cost stages on stale/short-circuit calls).
        parts = [f"n={n}"]
        for k, v in self._timing_stage_us.items():
            parts.append(f"{k}={v / n:.1f}us")
        for k in ("defog_run", "wiener", "cnn"):
            parts.append(f"{k}_count={self._timing_run_counts[k]}")
        print("TIMING " + " ".join(parts), file=sys.stderr)

    # ------------------------------------------------------------------ cascade helpers
    def _aruco_detect(self, gray: np.ndarray):
        """Detect marker 0 in a grayscale frame. Returns
        ``(image_points or None, num_rejected_candidates)``.

        The rejected count is what makes the cascade affordable: if ArUco
        saw zero polygon candidates the frame is featureless and Wiener
        deblur cannot manufacture marker structure from noise — skip it
        instead of paying the second adaptive-threshold scan (~50 ms with
        the widened win-size range tuned for HARD fog).
        """
        if self._detector is not None:
            corners, ids, rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )
        n_rejected = 0 if rejected is None else len(rejected)
        if ids is None or len(ids) == 0:
            return None, n_rejected
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                return np.asarray(c, dtype=np.float32).reshape(4, 2), n_rejected
        return None, n_rejected

    def _aruco_and_pose(
        self, gray: np.ndarray, stale: bool, source: str,
    ) -> Dict:
        """Run ArUco + IPPE_SQUARE solvePnP on a preprocessed grayscale frame.
        Returns the perception dict (with `source` tag) or
        ``{detected: False, stale}`` on miss/reject."""
        image_points, _ = self._aruco_detect(gray)
        if image_points is None:
            return {"detected": False, "stale": stale}
        return self._solve_pose_from_corners(image_points, stale, source)

    def _solve_pose_from_corners(
        self, image_points: np.ndarray, stale: bool, source: str,
    ) -> Dict:
        """IPPE_SQUARE pose + confidence on four corner pixels. Shared by the
        ArUco / deconv / CNN paths."""
        # solvePnPGeneric with IPPE_SQUARE returns both planar-PnP solutions.
        # solvePnP would silently pick one and hide the mirror-pose flip —
        # the #1 catastrophic-failure cause at low tilt.
        ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            self._object_points,
            image_points,
            self.intrinsics,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or errors is None or len(errors) == 0:
            return {"detected": False, "stale": stale}

        errors = np.asarray(errors, dtype=np.float64).reshape(-1)
        best = int(np.argmin(errors))
        rvec = np.asarray(rvecs[best], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[best], dtype=np.float64).reshape(3)
        err_min = float(errors.min())
        err_max = float(errors.max())
        ambiguous = (err_max / max(err_min, 1e-6)) < 1.5

        # IPPE_SQUARE silently emits NaN tvec when the four corners are
        # nearly collinear (near the FOV edge, extreme angles). Also reject
        # solutions where the marker is behind the camera or absurdly far.
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False, "stale": stale}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False, "stale": stale}

        proj, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, self.intrinsics, self.dist_coeffs
        )
        proj = proj.reshape(4, 2)
        reprojection_error = float(
            np.sqrt(np.mean(np.sum((proj - image_points) ** 2, axis=1)))
        )
        # Anything above 3 px is unreliable — treat as no detection so the KF
        # runs predict-only instead of being polluted.
        if reprojection_error > 3.0:
            return {"detected": False, "stale": stale}

        marker_area_px = float(cv2.contourArea(image_points.astype(np.float32)))
        area_score = min(1.0, marker_area_px / 5000.0)
        err_score = float(np.exp(-reprojection_error / 2.0))
        source_prior = self._SOURCE_PRIOR.get(source, 1.0)
        confidence = float(err_score * area_score * source_prior)

        return {
            "detected": True,
            "stale": stale,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
            "reprojection_error": reprojection_error,
            "confidence": confidence,
            "ambiguous": ambiguous,
            "source": source,
        }

    # ------------------------------------------------------------------ defog
    def _defog_dark_channel(self, rgb: np.ndarray) -> np.ndarray:
        """Single-image dehazing via dark-channel prior (He et al. 2009).

        Compute the dark-channel transmission map at 1/4 resolution, refine
        with a guided filter, upsample, then invert the haze blend
        `I = t * J + (1 - t) * A` for `J`. Targets ~3-5 ms on the eval CPU.

        Fog gate: on truly clean frames the dark-channel is ~0 everywhere,
        the transmission map is ~1 everywhere, and the algorithm reduces
        to J ≈ I — an expensive identity. Probe a sub-sampled minimum
        first and short-circuit when at least one pixel is genuinely dark.
        """
        # Fog-presence probe — 1/64 subsample, ~0.05 ms on uint8. A single
        # genuinely-dark pixel under any channel is sufficient evidence that
        # the frame isn't washed out by haze.
        if rgb[::8, ::8].min() < self._defog_skip_thresh:
            return rgb
        h, w = rgb.shape[:2]
        qw = max(1, w // self._defog_scale)
        qh = max(1, h // self._defog_scale)
        rgb_small = cv2.resize(rgb, (qw, qh), interpolation=cv2.INTER_AREA)
        rgb_small_f = rgb_small.astype(np.float32)

        # Atmospheric light A: brightest 0.1% pixels of the small dark channel
        # define the haze color we will subtract back out.
        dark_small = np.min(rgb_small_f, axis=2)
        dark_small = cv2.erode(dark_small, self._defog_kernel)
        flat = dark_small.ravel()
        n = flat.size
        k = max(1, n // 1000)
        top_idx = np.argpartition(flat, -k)[-k:]
        A = rgb_small_f.reshape(-1, 3)[top_idx].mean(axis=0)
        A = np.maximum(A, 1.0)  # avoid div-by-zero on a pitch-black frame

        # Transmission map at low res.
        norm = rgb_small_f / A
        dc_norm = np.min(norm, axis=2)
        dc_norm = cv2.erode(dc_norm, self._defog_kernel)
        t = (1.0 - self._defog_omega * dc_norm).astype(np.float32)

        # Refine with guided filter (or bilateral if ximgproc is unavailable).
        # Guide MUST be float [0, 1] — cv2.ximgproc.guidedFilter produces NaN
        # when uint8 guide is paired with a sub-unit eps (eps is in guide units).
        gray_small_f = (
            cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        )
        try:
            t = cv2.ximgproc.guidedFilter(
                guide=gray_small_f,
                src=t,
                radius=self._defog_gf_radius,
                eps=self._defog_gf_eps,
            )
        except AttributeError:
            t = cv2.bilateralFilter(t, 9, 0.1, 5)

        t_full = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
        t_full = np.clip(t_full, self._defog_t0, 1.0)[..., None]

        # Recover J = (I - A) / max(t, t0) + A. nan_to_num guards the cast
        # against pathological frames (all-black + degenerate A).
        rgb_f = rgb.astype(np.float32)
        J = (rgb_f - A) / t_full + A
        J = np.nan_to_num(J, nan=128.0, posinf=255.0, neginf=0.0)
        return np.clip(J, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ deblur
    def _deblur_wiener(self, gray: np.ndarray) -> np.ndarray:
        """Wiener deconvolution for the env's horizontal 1x3 box motion blur.

        Fallback-only — Wiener on clean frames rings the marker's quiet-zone
        border. Precomputed FFT of a centered kernel keeps this at ~2 ms.
        """
        # Pad/crop to the expected (480, 640) the precomputed kernel assumes.
        if gray.shape != (480, 640):
            return gray
        G = np.fft.rfft2(gray.astype(np.float32))
        K = self._wiener_K
        H = np.conj(K) / (np.abs(K) ** 2 + self._wiener_nsr)
        restored = np.fft.irfft2(G * H, s=gray.shape)
        return np.clip(restored, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ CNN fallback
    def _try_load_cnn(self) -> None:
        """Lazy-load the corner-detector ONNX. Missing model => stay disabled
        and let perceive() fall through cleanly. Warm-up forward primes
        cv2.dnn's buffer allocation so the first real call is hot."""
        model_path = Path(__file__).resolve().parent / "models" / "corner_detector.onnx"
        if not model_path.exists():
            return
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            dummy = np.zeros((1, 1, 72, 96), dtype=np.float32)
            net.setInput(dummy)
            net.forward()
            self._cnn_net = net
            self._cnn_ready = True
        except Exception as exc:  # noqa: BLE001 — non-fatal degradation
            import sys
            print(
                f"[Agent] CNN load failed ({exc}); perception will fall through "
                f"to detected=False on hard frames.",
                file=sys.stderr,
            )
            self._cnn_net = None
            self._cnn_ready = False

    def _cnn_corners(self, rgb_full: np.ndarray) -> Optional[np.ndarray]:
        """Predict marker corner pixels from a degraded frame. Returns a
        (4, 2) float32 array in TL/TR/BR/BL order matching ArUco's convention,
        or None when the visibility head says the marker is absent."""
        if self._cnn_net is None:
            return None
        try:
            gray = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY)
            inp = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
            inp_f = (inp.astype(np.float32) / 255.0).reshape(1, 1, 72, 96)
            self._cnn_net.setInput(inp_f)
            out = self._cnn_net.forward().reshape(-1)
            if out.size != 9 or not np.all(np.isfinite(out)):
                return None
            visible = float(out[8])
            if visible < 0.5:
                return None
            corners_norm = out[:8].reshape(4, 2)
            h, w = rgb_full.shape[:2]
            corners_px = np.empty((4, 2), dtype=np.float32)
            corners_px[:, 0] = corners_norm[:, 0] * float(w)
            corners_px[:, 1] = corners_norm[:, 1] * float(h)
            if not np.all(np.isfinite(corners_px)):
                return None
            return corners_px
        except Exception:  # noqa: BLE001
            return None

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
        # Skip duplicate frames: applying the same measurement twice falsely
        # shrinks the KF covariance. perceive() flags this via "stale".
        if detected and bool(perception.get("stale", False)):
            detected = False
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

        # CNN-source poses go through position update but not heading: a
        # corner-regression rvec is noisier than detection-source rvec, and
        # the heading filter is intentionally distrustful.
        if psi_meas is not None and perception.get("source") == "cnn":
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
            # Reset to a 5-D zero-velocity / zero-turn-rate prior with a
            # wide cov on position and looser priors on velocity/omega so
            # the filter can re-acquire when the marker comes back.
            self._kf.x = np.array(
                [float(anchor[0]), float(anchor[1]), 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            self._kf.P = np.diag(
                [4.0, 4.0, 1.0, 1.0, 0.1 ** 2]
            ).astype(np.float64)
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
    @staticmethod
    def _phase_from(phase_out) -> str:
        """Backwards-compat: ``decide()`` may return either a bare phase
        string (baseline) or a dict per contracts.md (planned). Accept both
        so this stage doesn't block the decide-stage upgrade.
        """
        if isinstance(phase_out, str):
            return phase_out
        if isinstance(phase_out, dict):
            phase = phase_out.get("phase")
            if isinstance(phase, str):
                return phase
        # Defensive fallback — unknown shape means we have no phase info.
        return PHASE_SEARCH

    # --- Tilt compensation (control.md item #6) ---
    # Standard altitude-control correction: when the drone is banking to
    # chase a lateral target, the body-z thrust contribution to world-z
    # collapses by cos(roll)*cos(pitch). Boosting the commanded thrust
    # by 1/tilt_factor maintains altitude under aggressive lateral tracking.
    # Floor at 0.5 (== ~60deg tilt) so we never multiply by a runaway number.
    TILT_COMP_FLOOR = 0.5

    # --- Active yaw control (control.md item #5) ---
    # Mod-pi error against the heading KF estimate. Gated on
    # phase_out.yaw_locked so decide() can hold the freeze until the
    # estimate is stable. Deadband + gain-scheduled + rate-limited by
    # phase - VTOL has weak yaw (k_Q/k_T = 0.020, high Izz, ~2s settle),
    # so we start yawing in APPROACH instead of waiting for touchdown.
    YAW_DEADBAND_RAD      = float(np.deg2rad(8.0))
    YAW_KP_APPROACH       = 1.0
    YAW_KP_DESCEND        = 0.5
    YAW_RATE_MAX_APPROACH = 1.5     # rad/s - comfortable yaw slew at altitude
    YAW_RATE_MAX_LAND     = 0.4     # rad/s - gentle near touchdown

    # --- Velocity feedforward (control.md item #2; 30-50% lateral RMS cut) ---
    # Cheap PD-style FF: the predicted body-frame velocity to track the boat
    # is added on top of the position PID as a small pitch/roll bias.
    # Reading the boat-velocity KF estimate when present, defaulting to zero
    # otherwise (baseline behavior preserved when KF is uninitialised).
    VEL_FF_GAIN_APPROACH = 0.15
    VEL_FF_GAIN_LAND = 0.7 * 0.15   # 0.105 — softer near touchdown
    POS_TO_VEL_GAIN = 0.5           # position-error → desired velocity gain

    # --- Soft-landing tunables (control.md item #4, scorer +5) ---
    SOFT_LAND_V_MAX = 0.6           # m/s — peak descent rate at altitude
    SOFT_LAND_H_TAU = 0.5           # m   — exponential profile time-constant
    SOFT_LAND_CRAWL_H = 0.10        # m   — switch to terminal crawl below this
    SOFT_LAND_CRAWL_VZ = -0.15      # m/s — terminal-crawl target
    SOFT_LAND_KP_VZ = 1.2           # P gain on vz error → normalized thrust
    SOFT_LAND_THRUST_FLOOR = -0.7   # min thrust — never command full free-fall

    def _soft_land_thrust(
        self,
        z_above: float,
        vz_world: float,
        target_descent_rate: Optional[float] = None,
    ) -> float:
        """Exponential soft-landing thrust law.

        ``vz_target = -v_max * (1 - exp(-h / tau))`` ramps from 0 m/s at the
        deck up to ``-v_max`` at altitude, with terminal-crawl override
        below ``SOFT_LAND_CRAWL_H``. A P loop on ``(vz_target - vz_world)``
        produces normalised thrust; floored at ``SOFT_LAND_THRUST_FLOOR`` so
        the controller never commands literal free-fall.

        Stays strictly below the scorer's ``< 1.0 m/s`` threshold for the
        +5 soft-landing bonus across the realistic LAND-phase altitude band
        (h ∈ [0, ~0.6] m given DESCEND_ALTITUDE_OK = 0.6).

        ``target_descent_rate`` (optional, from ``phase_out``) lets
        ``decide()`` cap the peak descent rate without coupling this
        controller to its policy.
        """
        v_max = self.SOFT_LAND_V_MAX
        if target_descent_rate is not None:
            try:
                tdr = float(target_descent_rate)
                if np.isfinite(tdr) and tdr > 0.0:
                    v_max = min(v_max, tdr)
            except (TypeError, ValueError):
                pass

        h = max(float(z_above), 0.0)
        vz_target = -v_max * (1.0 - np.exp(-h / max(self.SOFT_LAND_H_TAU, 1e-6)))
        if h < self.SOFT_LAND_CRAWL_H:
            vz_target = self.SOFT_LAND_CRAWL_VZ
        err_vz = vz_target - float(vz_world)
        thrust_raw = self.SOFT_LAND_KP_VZ * err_vz
        thrust_clipped = float(np.clip(thrust_raw, -1.0, 1.0))
        return max(thrust_clipped, self.SOFT_LAND_THRUST_FLOOR)

    def _control_setpoints(
        self, drone_state: Dict, boat_estimate: Dict, phase_out,
    ) -> Dict[str, float]:
        """Compute the high-level (thrust, roll, pitch, yaw_rate) setpoints.

        Shared between ``control()`` (which then mixes into per-motor
        throttles) and ``act_setpoint()`` (which returns these setpoints
        directly for the FC-compatible interface, +15 pts per
        ``teams/stick-the-landing/notes/contracts.md``).

        All four outputs are guaranteed finite — non-finite intermediates
        collapse to the safe-hover setpoint (0, 0, 0, 0).
        """
        phase = self._phase_from(phase_out)

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

        # .get() with SEARCH fallback so PHASE_ALTITUDE lookup never raises
        # if decide ever returns a phase we don't recognise (additive contract).
        target_z = float(target[2]) + self.PHASE_ALTITUDE.get(
            phase, self.PHASE_ALTITUDE[PHASE_SEARCH],
        )

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
        # Tilt compensation: divide commanded thrust by cos(roll)*cos(pitch)
        # so altitude does not sag when the drone is banking laterally.
        roll_att_for_tilt = float(drone_state["attitude"][0])
        pitch_att_for_tilt = float(drone_state["attitude"][1])
        tilt_factor = max(
            float(np.cos(roll_att_for_tilt) * np.cos(pitch_att_for_tilt)),
            self.TILT_COMP_FLOOR,
        )
        thrust_cmd = float(thrust_cmd) / tilt_factor

        # --- Velocity feedforward (control.md item #2) ---
        # Build a body-frame desired-velocity vector from the position error
        # and the KF boat-velocity estimate, then bias pitch/roll toward it.
        # Sign mapping matches the PID terms: +body_x → +pitch, +body_y → −roll.
        # Defaults preserve baseline when KF velocity is missing/non-finite.
        target_v_world_raw = boat_estimate.get("velocity")
        target_v_world = np.zeros(3, dtype=np.float64)
        if target_v_world_raw is not None:
            try:
                _tv = np.asarray(target_v_world_raw, dtype=np.float64).reshape(3)
                if np.all(np.isfinite(_tv)):
                    target_v_world = _tv
            except (TypeError, ValueError):
                pass
        v_des_world = err_world * self.POS_TO_VEL_GAIN + target_v_world
        v_des_body = R.T @ v_des_world
        ff_gain = (
            self.VEL_FF_GAIN_LAND if phase == PHASE_LAND
            else self.VEL_FF_GAIN_APPROACH
        )
        pitch_cmd = float(np.clip(pitch_cmd + ff_gain * float(v_des_body[0]), -1.0, 1.0))
        roll_cmd  = float(np.clip(roll_cmd  - ff_gain * float(v_des_body[1]), -1.0, 1.0))
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
        # LAND-phase thrust:
        # - Default (legacy bang-bang): full negative thrust pushes through
        #   the ground-effect cushion and the drone touches down, but with
        #   max_descent_velocity saturated — forfeits the soft-landing bonus.
        # - Soft profile (control.md #4 + contracts.md): exponential vz
        #   target tracked with a P loop, floored at -0.7, opt-in via
        #   phase_out.commit_soft. Scorer caps the +5 bonus at vz < 1.0 m/s.
        if phase == PHASE_LAND:
            commit_soft = False
            target_descent_rate = None
            if isinstance(phase_out, dict):
                commit_soft = bool(phase_out.get("commit_soft", False))
                target_descent_rate = phase_out.get("target_descent_rate")
            if commit_soft:
                thrust_cmd = self._soft_land_thrust(
                    z_above=float(pos[2] - target[2]),
                    vz_world=float(vel_world[2]),
                    target_descent_rate=target_descent_rate,
                )
            else:
                thrust_cmd = -1.0

        # --- Active yaw control (control.md item #5) ---
        # Baseline: yaw_rate_cmd = 0 (no active yaw). Opt-in path: when
        # decide() signals yaw_locked AND we have a heading estimate,
        # compute a mod-pi yaw error and command a P-controlled, rate-
        # limited yaw rate. Mod-pi because the landing condition is
        # fuselage-aligned-mod-pi (both 0 and pi pass).
        yaw_rate_cmd = 0.0
        yaw_locked = False
        yaw_target_override = None
        if isinstance(phase_out, dict):
            yaw_locked = bool(phase_out.get("yaw_locked", False))
            yaw_target_override = phase_out.get("yaw_target")
        boat_heading = boat_estimate.get("heading")
        if yaw_locked and boat_heading is not None:
            try:
                boat_yaw_f = float(boat_heading)
            except (TypeError, ValueError):
                boat_yaw_f = float("nan")
            if np.isfinite(boat_yaw_f):
                target_yaw = boat_yaw_f
                if yaw_target_override is not None:
                    try:
                        ty = float(yaw_target_override)
                        if np.isfinite(ty):
                            target_yaw = ty
                    except (TypeError, ValueError):
                        pass
                drone_yaw = float(drone_state["attitude"][2])
                yaw_err = _wrap_mod_pi(target_yaw - drone_yaw)
                if abs(yaw_err) >= self.YAW_DEADBAND_RAD:
                    kp_yaw = (
                        self.YAW_KP_DESCEND if phase in (PHASE_DESCEND, PHASE_LAND)
                        else self.YAW_KP_APPROACH
                    )
                    rate_max = (
                        self.YAW_RATE_MAX_APPROACH if phase == PHASE_APPROACH
                        else self.YAW_RATE_MAX_LAND
                    )
                    yaw_rate_cmd = float(np.clip(kp_yaw * yaw_err, -rate_max, rate_max))

        # NaN/Inf guard. Per contracts.md the four setpoints MUST be finite
        # — collapse any anomaly to the safe-hover setpoint rather than
        # propagating poison through attitude_ctrl / the FC link.
        sp = {
            "thrust": float(thrust_cmd),
            "roll": float(roll_cmd),
            "pitch": float(pitch_cmd),
            "yaw_rate": float(yaw_rate_cmd),
        }
        for k, v in sp.items():
            if not np.isfinite(v):
                sp[k] = 0.0
        return sp

    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase_out
    ) -> np.ndarray:
        # phase_out may be a bare phase string (baseline) or the contract
        # dict (decide-stage upgrade). _control_setpoints normalises both.
        sp = self._control_setpoints(drone_state, boat_estimate, phase_out)

        # Hand the high-level (thrust, roll, pitch, yaw_rate) setpoints to
        # the stock attitude controller, which mixes them into per-motor
        # throttles using the drone spec's geometry and propeller params.
        return self.attitude_ctrl(
            drone_state,
            roll_des=sp["roll"],
            pitch_des=sp["pitch"],
            yaw_rate_des=sp["yaw_rate"],
            thrust_norm=sp["thrust"],
        )

    # ------------------------------------------------------------------ FC-compatible setpoint API
    def act_setpoint(self, obs: Dict):
        """Flight-controller-compatible 4-DoF setpoint output.

        Returns ``(thrust_norm, roll, pitch, yaw_rate)`` — the same four
        setpoints ``control()`` already computes internally. Per
        ``evaluation/evaluate.py:146-163``, the mere presence of this
        method earns the +15 FC-compatibility bonus on every LANDED run.

        Runs the same perceive → estimate → decide → control pipeline as
        ``act()`` but stops at the setpoint stage — the evaluator owns the
        attitude controller in setpoint mode.
        """
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state)
        self._last_estimate = boat_estimate
        phase_out = self.decide(state, boat_estimate, battery, time_s)
        # Keep self.phase in sync with the act() path so external inspectors
        # see consistent phase regardless of which entrypoint the eval uses.
        self.phase = self._phase_from(phase_out)

        sp = self._control_setpoints(state, boat_estimate, phase_out)

        # Final NaN guard — if anything upstream still leaked non-finite
        # (shouldn't, _control_setpoints already clamps), fall back to
        # the safe-hover setpoint (zero attitude, hover thrust = 0.0).
        thrust = sp["thrust"] if np.isfinite(sp["thrust"]) else 0.0
        roll   = sp["roll"]   if np.isfinite(sp["roll"])   else 0.0
        pitch  = sp["pitch"]  if np.isfinite(sp["pitch"])  else 0.0
        yaw    = sp["yaw_rate"] if np.isfinite(sp["yaw_rate"]) else 0.0
        return (float(thrust), float(roll), float(pitch), float(yaw))


# Module-level entry point. evaluation/evaluate.py imports the module
# dynamically and looks for a callable named `make_agent` (preferred) or a
# class named `Agent`. We support both for convenience.
Agent = BaselineAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> BaselineAgent:
    return BaselineAgent(drone_spec)
