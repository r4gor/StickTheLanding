# Perception stage playbook

**Owner suggestion:** one of the CS devs. ~2-3 hours of work for ~80% of available gains.

**Goal:** halve tvec jitter, eliminate IPPE_SQUARE mirror-pose failures, survive HARD fog/blur.

**Stays stateless and pure CV.** All temporal/filtering work belongs in `estimate`.

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
Saddle-point fitting on all 4 corners before PnP. **Roughly halves tvec jitter at altitude.** Free 2× on attitude accuracy. OpenCV-contrib only (already in requirements).

### 2. IPPE_SQUARE disambiguation — 30 min, biggest safety win
The current `solvePnP(IPPE_SQUARE)` silently returns *one* of two valid solutions; the "mirror" pose is the #1 cause of catastrophic landing failures at low tilt.

Switch to:
```python
ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
    obj_pts, image_points, K, None,
    flags=cv2.SOLVEPNP_IPPE_SQUARE,
)
# pick the lower-reprojection-error solution
i = int(np.argmin(errors))
rvec, tvec, err = rvecs[i], tvecs[i], float(errors[i])
ambiguous = (errors.max() / max(errors.min(), 1e-6)) < 1.5
```
Cross-check: require `tvec[2] > 0` (marker in front of camera) and the marker's local +Z axis pointing roughly toward the camera. Emit `ambiguous=True` so estimate can inflate covariance.

### 3. Reprojection-error confidence — 15 min, unlocks downstream gating
```python
proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
err_px = float(np.linalg.norm(image_points - proj.reshape(4, 2), axis=1).mean())
confidence = float(np.exp(-err_px / 2.0))   # 1.0 = perfect, ~0.1 at 5 px
detected = detected and (err_px < 3.0)      # hard reject above 3 px
```
Put both `reprojection_error` and `confidence` in the output dict (see `contracts.md`).

### 4. CLAHE for HARD fog — 20 min, scenario-specific
Beer-Lambert fog is a **local-contrast killer**, not global. CLAHE wins over gamma/gray-stretch.
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)   # ~1-2 ms on 720p
```
Run unconditionally — the cost is in the budget. (Optional: skip when initial detection succeeds; retry with CLAHE on failure.)

### 5. Detector params for motion blur — 30 min
For HARD's 10 Hz frames:
```python
params.adaptiveThreshWinSizeMax = 53
params.polygonalApproxAccuracyRate = 0.05
params.minMarkerPerimeterRate = 0.02
```
If detection fails on full-res, retry on `cv2.resize(gray, None, fx=0.5, fy=0.5)` — blur kernel shrinks proportionally. PnP scales linearly.

---

## What to skip

| Idea | Why skip |
|---|---|
| `refineDetectedMarkers` | Designed for marker boards, not single markers |
| Deblurring (Wiener, deep) | Too slow + brittle in 20 ms budget |
| Multi-frame aggregation inside `perceive()` | Belongs in `estimate()` — keeps perception stateless and debuggable |
| Neural detectors | Overkill; ArUco at 0.8 m is easy modality |

---

## Acceptance checklist

- [ ] AprilTag refinement enabled
- [ ] `solvePnPGeneric` returns two solutions; lower-error chosen
- [ ] `reprojection_error` and `confidence` in output
- [ ] `ambiguous` flag emitted when needed
- [ ] CLAHE applied (HARD parity)
- [ ] Detector params tuned for blur
- [ ] All optional fields default to None; type-hinted
- [ ] Unit test: a mock image with a marker → known tvec to ±0.05 m

**Total budget:** ~2 hours. Anything above that is diminishing returns vs working on `estimate` or `control`.
