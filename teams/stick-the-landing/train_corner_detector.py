"""Host-side training pipeline for the CNN corner-regressor fallback used in
agent.py's perception cascade (tier C).

Two modes:

    # Step 1 — collect labelled (image, corners, visibility) tuples by
    # spinning up BoatLandingEnv across all scenarios + many seeds. Runs
    # inside Docker because it needs PyBullet + BoatLandingEnv. No torch
    # required.
    ./docker/run-local.sh python teams/stick-the-landing/train_corner_detector.py \
        --mode collect --out teams/stick-the-landing/models/corner_data.npz \
        --num-frames 20000

    # Step 2 — train the model. Runs on the host (Mac), needs torch + onnx.
    # Reads the .npz, fits the tiny CNN, exports an ONNX that cv2.dnn can load.
    python teams/stick-the-landing/train_corner_detector.py \
        --mode train --data teams/stick-the-landing/models/corner_data.npz \
        --out teams/stick-the-landing/models/corner_detector.onnx \
        --epochs 40

Design choices and rationale: see teams/stick-the-landing/notes/perceive.md.
The CNN's job is to recover marker corner pixel coordinates on frames where
ArUco's polygon-decode pipeline fails (heavy fog × motion blur × noise on
HARD). Output is fed back into the same solvePnP + confidence machinery so
the downstream KF doesn't change.

Labels are produced by running ArUco on the CLEAN render and reusing the
detected corners on degraded copies of the same frame — sidesteps the
boat-frame-to-pixel projection math and guarantees label ordering matches
cv2.aruco.detectMarkers' TL/TR/BR/BL convention.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (REPO_ROOT, HERE):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ---------------------------------------------------------------------------
# CONSTANTS — must match agent.py's CNN inference path.
# ---------------------------------------------------------------------------
INPUT_W = 96
INPUT_H = 72
NUM_OUTPUTS = 9  # 8 corner coords + 1 visibility

CAMERA_W = 640
CAMERA_H = 480


# ---------------------------------------------------------------------------
# DATA COLLECTION
# ---------------------------------------------------------------------------

def _detect_clean_corners(frame_rgb: np.ndarray):
    """Run a clean ArUco detection on a freshly rendered frame to get the
    ground-truth pixel corners. Returns (4, 2) float32 in TL/TR/BR/BL order,
    or None if marker 0 is not detectable."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    params = cv2.aruco.DetectorParameters()
    try:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    except AttributeError:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    try:
        detector = cv2.aruco.ArucoDetector(adict, params)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
    if ids is None or len(ids) == 0:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == 0:
            return np.asarray(c, dtype=np.float32).reshape(4, 2)
    return None


def _render_frame(env, drone_pos: np.ndarray, drone_attitude: np.ndarray) -> np.ndarray:
    """Force the env to render the camera at a chosen drone pose. Bypasses
    the normal flight loop so we can saturate marker geometry."""
    import pybullet as p
    # Reset drone sim state.
    env.drone_sim.reset(np.asarray(drone_pos, dtype=np.float64),
                        np.asarray(drone_attitude, dtype=np.float64))
    state = env.drone_sim.get_state()
    # Sync PyBullet drone body for camera occlusion math (the env's drone is
    # a kinematic body for collision; for camera it only needs pose).
    p.resetBasePositionAndOrientation(
        env.drone_id,
        state.position.tolist(),
        state.quaternion.tolist(),
        physicsClientId=env.client,
    )
    return env._render_camera(state.position, state.quaternion)


def _apply_aug(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Layer extra degradations on top of whatever the scenario already
    applied. Bounds chosen above the worst observed scenario values
    (HARD: fog 0.10, blur 3, noise 0.05) to give the CNN headroom."""
    from boat_landing.camera import apply_fog, apply_motion_blur, apply_noise

    out = frame
    if rng.random() < 0.7:
        fog = float(rng.uniform(0.0, 0.15))
        if fog > 0.005:
            # Pick a plausible altitude for the fog blend strength — the
            # higher the value the more aggressive the wash-out.
            out = apply_fog(out, fog, float(rng.uniform(2.0, 8.0)))
    if rng.random() < 0.5:
        k = int(rng.choice([0, 3, 5]))
        if k >= 3:
            out = apply_motion_blur(out, k)
    if rng.random() < 0.7:
        sigma = float(rng.uniform(0.0, 0.08))
        if sigma > 0.005:
            out = apply_noise(out, sigma, np.random.default_rng(int(rng.integers(0, 2**31))))
    if rng.random() < 0.15:
        # Random black/grey rectangle occlusion. Roughly mimics partial
        # sensor obstruction. Keep marker visible if the patch doesn't
        # cover its center, but the label keeps it visible (the CNN must
        # still localize partially-occluded corners).
        h, w = out.shape[:2]
        rh = int(rng.uniform(0.05, 0.30) * h)
        rw = int(rng.uniform(0.05, 0.30) * w)
        y0 = int(rng.integers(0, max(1, h - rh)))
        x0 = int(rng.integers(0, max(1, w - rw)))
        out = out.copy()
        out[y0:y0+rh, x0:x0+rw] = int(rng.integers(0, 80))
    return out


def collect(out_path: Path, num_frames: int, seed: int) -> None:
    """Walk through scenarios, sample diverse drone poses, render frames,
    label via ArUco on the clean render, save (image, corners_normalized,
    visibility) as a single .npz."""
    from boat_landing.env import BoatLandingEnv

    scenarios = ["easy", "medium", "hard", "recovery"]
    rng = np.random.default_rng(seed)

    images = np.empty((num_frames, INPUT_H, INPUT_W), dtype=np.uint8)
    corners_norm = np.zeros((num_frames, 8), dtype=np.float32)
    vis = np.zeros((num_frames, 1), dtype=np.float32)
    saved = 0
    attempts = 0
    progress_step = max(1, num_frames // 20)

    # One env per scenario, reused across many samples. Creating multiple
    # BoatLandingEnv objects in sequence trips PyBullet's DIRECT client
    # lifecycle — close()+connect() can return a stale client. Instead,
    # call env.reset(seed=...) periodically to re-randomize the boat init.
    per_scenario = max(1, num_frames // len(scenarios))
    resample_every = 40

    for scenario in scenarios:
        if saved >= num_frames:
            break
        scen_path = REPO_ROOT / "scenarios" / f"{scenario}.yaml"
        env = BoatLandingEnv(str(scen_path))
        try:
            env.reset(seed=int(rng.integers(0, 2**31)))
            samples_target = saved + min(per_scenario, num_frames - saved)
            local_count = 0
            while saved < samples_target and saved < num_frames:
                attempts += 1
                local_count += 1
                if local_count % resample_every == 0:
                    env.reset(seed=int(rng.integers(0, 2**31)))
                # Step the boat to a randomized time so the platform pose
                # varies across samples.
                t_skip = float(rng.uniform(0.0, 20.0))
                env.boat.step(t_skip)

                # Drone pose: above the boat, within camera FOV reach.
                bp = env.boat.position
                offset = rng.normal(0.0, 1.0, size=2) * 1.5
                drone_xy = bp[:2] + offset
                drone_z = float(rng.uniform(1.5, 9.0))
                drone_pos = np.array([drone_xy[0], drone_xy[1], drone_z])

                # Mostly level, occasional small tilt.
                roll = float(rng.normal(0.0, np.deg2rad(5.0)))
                pitch = float(rng.normal(0.0, np.deg2rad(5.0)))
                yaw = float(rng.uniform(-np.pi, np.pi))
                attitude = np.array([roll, pitch, yaw])

                clean = _render_frame(env, drone_pos, attitude)

                # Label from the clean detection.
                gt_corners = _detect_clean_corners(clean)
                if gt_corners is None:
                    # Marker not visible / undecodable on clean render — drop
                    # this sample. We don't fabricate labels we can't trust.
                    continue

                # Apply augmentations (independent extra degradations).
                degraded = _apply_aug(clean, rng)

                # Downscale to model input size.
                gray_full = cv2.cvtColor(degraded, cv2.COLOR_RGB2GRAY)
                gray_small = cv2.resize(
                    gray_full, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA
                )

                # Normalized labels in [0, 1] (full-resolution pixel
                # coordinates / camera dims so the network output maps back
                # to image space at inference time).
                norm = np.empty((4, 2), dtype=np.float32)
                norm[:, 0] = gt_corners[:, 0] / float(CAMERA_W)
                norm[:, 1] = gt_corners[:, 1] / float(CAMERA_H)

                images[saved] = gray_small
                corners_norm[saved] = norm.reshape(-1)
                in_frame = bool(
                    np.all(norm[:, 0] > -0.01)
                    and np.all(norm[:, 0] < 1.01)
                    and np.all(norm[:, 1] > -0.01)
                    and np.all(norm[:, 1] < 1.01)
                )
                vis[saved, 0] = 1.0 if in_frame else 0.0
                saved += 1
                if saved % progress_step == 0:
                    print(
                        f"  collected {saved}/{num_frames} "
                        f"(attempts={attempts}, hit_rate={saved/attempts:.2f})",
                        file=sys.stderr,
                    )
        finally:
            env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        corners=corners_norm,
        visibility=vis,
    )
    print(
        f"saved {saved} frames to {out_path} "
        f"(visible: {int(vis.sum())}, attempts: {attempts})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# MODEL + TRAINING (PyTorch — host-side only)
# ---------------------------------------------------------------------------

def _build_model():
    import torch
    import torch.nn as nn

    class CornerNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 4 stride-2 convs, BN, ReLU. Tiny: ~110K params at 96x72 input.
            def block(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            self.backbone = nn.Sequential(
                block(1, 16),    # 96x72 -> 48x36
                block(16, 32),   # 48x36 -> 24x18
                block(32, 64),   # 24x18 -> 12x9
                block(64, 64),   # 12x9  -> 6x5  (rounded)
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, NUM_OUTPUTS),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x).flatten(1)
            return self.head(x)

    return CornerNet()


def train(data_path: Path, out_path: Path, epochs: int, batch: int, lr: float) -> None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print(
            "error: torch not installed on host. Install with `pip install torch onnx`"
            " then re-run --mode train.",
            file=sys.stderr,
        )
        sys.exit(2)

    data = np.load(data_path)
    images = data["images"]            # (N, H, W) uint8
    corners = data["corners"]          # (N, 8) float32 [0,1]
    visibility = data["visibility"]    # (N, 1) float32

    # Train/val split — last 10% reserved for validation.
    N = images.shape[0]
    split = int(N * 0.9)
    perm = np.random.RandomState(0).permutation(N)
    tr_idx, va_idx = perm[:split], perm[split:]

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}", file=sys.stderr)

    def to_tensors(idx):
        x = torch.from_numpy(images[idx]).float().unsqueeze(1) / 255.0
        y_corners = torch.from_numpy(corners[idx]).float()
        y_vis = torch.from_numpy(visibility[idx]).float()
        return x, y_corners, y_vis

    x_tr, y_tr, v_tr = to_tensors(tr_idx)
    x_va, y_va, v_va = to_tensors(va_idx)

    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(x_tr.shape[0])
        total = 0.0
        for start in range(0, x_tr.shape[0], batch):
            sel = idx[start:start + batch]
            xb = x_tr[sel].to(device)
            yc = y_tr[sel].to(device)
            yv = v_tr[sel].to(device)
            pred = model(xb)
            pred_c = pred[:, :8]
            pred_v = pred[:, 8:9]
            loss_v = F.binary_cross_entropy(pred_v, yv)
            # Mask corner loss by visibility — don't penalize predictions on
            # invisible-marker frames (CNN output is irrelevant there).
            loss_c = (F.l1_loss(pred_c, yc, reduction="none") * yv).mean()
            loss = loss_v + loss_c
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred = model(x_va.to(device))
            pred_c = pred[:, :8]
            pred_v = pred[:, 8:9]
            l_v = F.binary_cross_entropy(pred_v, v_va.to(device)).item()
            l_c = (F.l1_loss(pred_c, y_va.to(device), reduction="none") * v_va.to(device)).mean().item()
            # Visible-corner mean pixel error.
            mask = v_va.to(device).squeeze().bool()
            err_px = ((pred_c[mask] - y_va.to(device)[mask]).abs().mean()
                      * float(CAMERA_W) ).item() if mask.any() else float("nan")
        print(
            f"epoch {epoch+1:3d}/{epochs}  train={total/x_tr.shape[0]:.4f}  "
            f"val_v={l_v:.4f}  val_c={l_c:.4f}  visible_mean_px_err={err_px:.2f}",
            file=sys.stderr,
        )

    # Export ONNX. opset 11 is the safest target for cv2.dnn 4.x.
    model.eval()
    dummy = torch.zeros(1, 1, INPUT_H, INPUT_W, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        str(out_path),
        opset_version=11,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=None,
    )
    print(f"exported ONNX to {out_path}", file=sys.stderr)

    # cv2.dnn cross-check: load the model back and verify a forward pass
    # produces near-identical output to the PyTorch model on the same input.
    try:
        net = cv2.dnn.readNetFromONNX(str(out_path))
        np_in = dummy.cpu().numpy()
        net.setInput(np_in)
        cv_out = net.forward().reshape(-1)
        with torch.no_grad():
            pt_out = model(dummy.cpu()).numpy().reshape(-1)
        max_diff = float(np.max(np.abs(cv_out - pt_out)))
        print(f"cv2.dnn vs torch max abs diff on dummy: {max_diff:.6f}", file=sys.stderr)
        if max_diff > 1e-3:
            print("warning: cv2.dnn output diverges from torch — verify opset compatibility.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"cv2.dnn cross-check failed (non-fatal): {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "train"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None,
                        help="(train mode) npz produced by --mode collect.")
    parser.add_argument("--num-frames", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.out, args.num_frames, args.seed)
    else:
        if args.data is None:
            print("error: --data is required in train mode", file=sys.stderr)
            return 2
        train(args.data, args.out, args.epochs, args.batch, args.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
