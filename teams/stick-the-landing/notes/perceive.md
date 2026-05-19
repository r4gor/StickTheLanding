# Perception stage playbook

**Owner suggestion:** one of the CS devs. ~2-3 hours of work for ~80% of available gains.

**Goal:** halve tvec jitter, eliminate IPPE_SQUARE mirror-pose failures, survive HARD fog/blur.

**Stays stateless and pure CV.** All temporal/filtering work belongs in `estimate`.

> **Research validation (2026-05-18):** All four major moves here (AprilTag refinement, IPPE_SQUARE disambiguation, reprojection-error confidence, CLAHE) are confirmed against PX4 `landing_target_estimator`, the IPPE paper (Collins et al.), and the OpenCV ArUco ambiguity issue tracker. PX4 ships a 2D constant-velocity KF on top of exactly this perception pipeline ([PX4 LandingTargetEstimator.cpp](https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/landing_target_estimator/LandingTargetEstimator.cpp)). The mirror-pose flip is a well-documented failure mode at low tilt ([arXiv 1909.11888](https://arxiv.org/pdf/1909.11888), [opencv answers](https://answers.opencv.org/question/123375/aruco-z-axis-flipping-perspective/)). CLAHE is benchmarked at ~14.5% accuracy lift under poor illumination ([defogging benchmark arXiv 2510.03906](https://arxiv.org/html/2510.03906v1)).

---

## Scenario reality check (from `scenarios/*.yaml` — actual, not README)

| | EASY | MEDIUM | HARD |
|---|---|---|---|
| Camera fps | every step (≈50 Hz) | **25 Hz** | **15 Hz** |
| Motion blur kernel | 0 | 0 | **3 px** (horizontal box blur) |
| Fog density | 0 | 0.03 | **0.10** (Beer-Lambert, blends toward gray) |
| Noise σ | 0 | 0.02 | 0.05 |
| Occlusion prob | 0 | 0 | **0.03/s** (5% per second of blackout chance, blackout lasts `occlusion_duration` s) |

HARD is meaningfully gentler than the README suggested (15 Hz, not 10; fog 0.10, not 0.15). Don't over-engineer.

---

## Baseline (what's there)

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
corners, ids, _ = cv2.aruco.ArucoDetector(DICT_5X5_100).detectMarkers(gray)
ok, rvec, tvec = cv2.solvePnP(obj_pts, image_points, K, None, IPPE_SQUARE)
return {"detected": bool, "tvec": (3,), "rvec": (3,), "image_points": (4,2)}
```

No sub-pixel refinement. Single PnP solution (mirror-pose risk). No confidence.

---

## Ranked work, in order

### 1. AprilTag corner refinement — 10 min, biggest accuracy win
```python
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
# fallback: CORNER_REFINE_SUBPIX if APRILTAG flag missing
detector = cv2.aruco.ArucoDetector(dictionary, params)
```
Saddle-point fitting on all 4 corners before PnP. **Roughly halves tvec jitter at altitude.** Free 2× on attitude accuracy. OpenCV-contrib only (already in requirements). Sub-pixel corner accuracy dominates pose error in IPPE ([IPPE paper](https://encov.ip.uca.fr/publications/pubfiles/2014_Collins_etal_IJCV_plane.pdf)).

### 2. IPPE_SQUARE disambiguation — 30 min, biggest safety win
The current `solvePnP(IPPE_SQUARE)` silently returns *one* of two valid solutions; the "mirror" pose is the #1 cause of catastrophic landing failures at low tilt.

Switch to:
```python
ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
    obj_pts, image_points, K, None,
    flags=cv2.SOLVEPNP_IPPE_SQUARE,
)
i = int(np.argmin(errors))
rvec, tvec, err = rvecs[i], tvecs[i], float(errors[i])
ambiguous = (errors.max() / max(errors.min(), 1e-6)) < 1.5
```
Cross-check: require `tvec[2] > 0` (marker in front of camera) and the marker's local +Z axis pointing roughly toward the camera. Emit `ambiguous=True` so estimate can inflate covariance.

### 3. Confidence + reprojection error — 15 min, unlocks downstream gating
Two refinements vs draft-1:

```python
# Reprojection error
proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
err_px = float(np.linalg.norm(image_points - proj.reshape(4, 2), axis=1).mean())

# Confidence weighted by both reprojection AND marker pixel area
# (pose error ∝ 1/pixel-size — see CEUR Vol-2744 ArUco accuracy)
marker_area_px = cv2.contourArea(image_points.astype(np.float32))
area_score = min(1.0, marker_area_px / 5000.0)   # ~5000 px² at altitude 5 m for 0.8 m marker
err_score  = np.exp(-err_px / 2.0)
confidence = float(err_score * area_score)

detected = detected and (err_px < 3.0)   # hard reject above 3 px reproj
```
Reference: [CEUR Vol-2744 ArUco accuracy study](https://ceur-ws.org/Vol-2744/short14.pdf) — pose error scales quadratically with marker pixel size.

### 4. CLAHE for HARD fog — 20 min, scenario-specific
Beer-Lambert fog is a local-contrast killer, not global. CLAHE wins over gamma/gray-stretch ([defogging benchmark arXiv 2510.03906](https://arxiv.org/html/2510.03906v1)).
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)   # ~1-2 ms on 720p
```
Run unconditionally — the cost is well in budget.

### 5. Detector params for motion blur — 30 min
For HARD's 15 Hz frames with 3 px horizontal blur:
```python
params.adaptiveThreshWinSizeMax = 53
params.polygonalApproxAccuracyRate = 0.05
params.minMarkerPerimeterRate = 0.02
```
If detection fails on full-res, retry on `cv2.resize(gray, None, fx=0.5, fy=0.5)` — blur kernel shrinks proportionally. PnP cost scales linearly.

### 6. Cheap deblur preprocessing for HARD (optional, ~30 min)
The web research surfaced that **deblurring lifts AprilTag detection from 32% to 80%+ on blurred frames** ([Ghost-DeblurGAN arXiv 2109.03379](https://ar5iv.labs.arxiv.org/html/2109.03379)). The deep nets are too slow for 20 ms, but a **classical unsharp mask** is cheap:
```python
# Apply only when detection fails on raw + CLAHE
sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (5, 5), 0), -0.5, 0)
```
Try this in the retry chain after CLAHE before downscaling.

---

## Cascade for HARD/recovery survivability (Tip #5 follow-up)

After the first six items landed, three additional perception techniques
from `knowledge-base.md` Tip #5 went in: dark-channel defog, Wiener motion-
blur deconvolution, and a CNN corner-regressor fallback. They are stacked
as a cascade so the cost only adds up on frames where the cheaper path
failed. The "deep deblurring / neural detectors are too slow" entries in
the original Skip table are superseded by the gates described below.

```
rgb → dark_channel_defog → gray → CLAHE → ArUco              (source="aruco")
                          (miss + n_rejected > 0)
                          → Wiener deblur → ArUco             (source="deconv")
                          (still miss)
                          → CNN corners → solvePnP            (source="cnn")
                          (still miss)
                          → {detected: False}
```

**Stale-frame short-circuit.** Identical-byte frames (camera fps below
50 Hz repeats the previous frame) return the previous perception dict
with `stale=True`. Drops 0.4 ms/call on cached frames; estimate() still
ignores stale measurements as before.

**Per-source confidence prior.** Multiplied into `confidence`:
`{aruco:1.0, deconv:0.7, cnn:0.5}`. estimate() already scales the KF R
matrix by `reprojection_error` and the heading filter gates on
`confidence > 0.5`. CNN-source detections additionally bypass the heading
update (corner-regression yaw is noisier than detection yaw).

### A. Dark-channel defog (always-on)

He et al. 2009 dehazing. Computes the transmission map at 1/4 resolution
(160×120), refines with `cv2.ximgproc.guidedFilter`, upsamples, inverts
`I = t·J + (1−t)·A` for `J`. Compounds with CLAHE downstream.

Gotcha: `guidedFilter` produces NaN when the guide is uint8 and eps is
sub-unit. Pass the guide as float32 [0,1] and keep eps=1e-3.

- Latency: ~4 ms/frame on the eval container.
- Fallback if `cv2.ximgproc` is missing: `cv2.bilateralFilter`.
- Tunables: `_defog_omega=0.95`, `_defog_t0=0.1`, `_defog_scale=4`,
  `_defog_gf_radius=15`, `_defog_gf_eps=1e-3`.

### B. Wiener deconvolution (fallback-only, gated)

Horizontal 1×3 box-blur deconvolution via numpy rfft2/irfft2, kernel
precomputed in `__init__`. Skipped when `n_rejected == 0` (no polygon
candidates exist → deblur cannot synthesize structure from noise), and
gated by a 12 ms total-cascade budget (so heavy noise frames don't
double the ArUco scan cost).

- Latency: ~2 ms/call. Only fires when ArUco's first pass saw quad
  candidates but couldn't decode marker 0.
- Rationale for fixed kernel: `boat_landing/camera.py:147` only ever
  applies a horizontal box of width 0 or 3 px. A misaligned guess on
  clean frames would ring the marker quiet-zone border, which is why
  this path is gated to runs where ArUco already failed.
- Tunables: `_wiener_nsr=0.01`. Kernel is centered (zero-phase) so the
  deconvolution preserves pixel alignment.

### C. CNN corner regressor (fallback-only)

A tiny CNN (96×72 grayscale input, ~110K params, 9 sigmoid outputs:
4×(x,y) + visibility) lazy-loaded from
`teams/stick-the-landing/models/corner_detector.onnx` via `cv2.dnn`.
Predicts the 4 marker pixel corners, which feed back into the same
`solvePnP(IPPE_SQUARE)` + confidence pipeline. Missing ONNX file is
non-fatal — the cascade falls through to `{detected: False}` cleanly.

- Backend: `cv2.dnn.readNetFromONNX`. No new requirements.txt entries.
- Warm-up: dummy `net.forward()` in `__init__` to allocate buffers so
  the first real call doesn't pay 50+ ms of one-time allocation cost.
- Latency target on the eval CPU (OpenBLAS, no MKL): ~10 ms.
- Invocation gate: only runs after both ArUco passes missed, and only
  if the cascade has < 15 ms elapsed (skip on heavy-noise frames where
  ArUco alone burned the budget).

#### Training (`train_corner_detector.py`)

Two-step procedure, host-side training:

```bash
# 1) Inside Docker — no torch needed. Renders frames from BoatLandingEnv,
#    labels each with ArUco corners on the CLEAN render, then writes
#    (image, corners, visibility) tuples for degraded copies of the
#    same frame. Sidesteps the boat-to-pixel projection math.
./docker/run-local.sh python teams/stick-the-landing/train_corner_detector.py \
    --mode collect \
    --out teams/stick-the-landing/models/corner_data.npz \
    --num-frames 20000

# 2) On the Mac with PyTorch installed. Trains the tiny CNN, exports
#    ONNX, runs a cv2.dnn cross-check against the torch model.
uv run python teams/stick-the-landing/train_corner_detector.py \
    --mode train \
    --data teams/stick-the-landing/models/corner_data.npz \
    --out  teams/stick-the-landing/models/corner_detector.onnx \
    --epochs 40
```

Augmentation goes ABOVE worst observed scenario values (fog up to 0.15,
blur kernels {0,3,5}, noise σ up to 0.08, occlusion patches up to 30%
of the frame) so the CNN has margin for private eval scenarios.

Training time on a Mac: 1–2 hours CPU/MPS for 40 epochs at ~20K frames.
The model is small enough that overfitting is more of a risk than
under-fitting; the train/val L1 gap is the metric to watch.

## What to skip

| Idea | Why skip |
|---|---|
| `refineDetectedMarkers` | Designed for marker boards, not single markers |
| Multi-frame aggregation inside `perceive()` | Belongs in `estimate()` — keeps perception stateless |

---

## NaN-safety reminder

If `tvec` or `rvec` contains NaN, **return `"detected": False`** — never pass NaN downstream. The baseline already guards `(np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec)))`. Keep it. Per `scorer.py:128-143`, NaN in any scored field forces worst-case scoring.

---

## Acceptance checklist

- [x] AprilTag refinement enabled
- [x] `solvePnPGeneric` returns two solutions; lower-error chosen
- [x] `reprojection_error` and `confidence` (area-weighted) in output
- [x] `ambiguous` flag emitted when needed
- [x] CLAHE applied unconditionally
- [x] Detector params tuned for blur
- [x] All optional fields default to None; type-hinted
- [x] Unit test: synthetic image with known marker → tvec to ±0.05 m
- [x] No NaN ever in output dict

### Cascade additions (Tip #5)

- [x] Dark-channel defog always-on before CLAHE
- [x] Wiener motion-blur deconv on fallback path, gated by `n_rejected > 0`
- [x] CNN corner regressor: inference wired via `cv2.dnn`, graceful when ONNX missing
- [x] Stale-frame short-circuit caches previous perception dict
- [x] Per-source confidence prior (aruco/deconv/cnn) + heading-skip for CNN
- [x] Unit tests: defog recovers fogged frame, deconv recovers blurred frame, cascade skips Wiener on featureless frames
- [ ] `corner_detector.onnx` trained and shipped (deferred — see train_corner_detector.py)

**Total budget:** ~2 hours base + ~3 hours for Tip #5 cascade. Anything above that is diminishing returns vs working on `estimate` or `control`.
