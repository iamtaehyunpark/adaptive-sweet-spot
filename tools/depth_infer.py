#!/usr/bin/env python3
"""Run one metric depth model on a capture; writes <capture>/<model>.npz.

Each model lives in its own environment (their dependencies conflict):

    .venvs/moge2/bin/python tools/depth_infer.py moge2 assets/captures/<name>
    .venvs/da3/bin/python   tools/depth_infer.py da3   assets/captures/<name>

Output npz: depth (H, W) z-depth in meters at capture resolution, fx, fy (px),
points (H, W, 3) camera-frame meters (x right, y down, z forward), focal_source.

Only numpy, cv2 and torch are used here, so this file runs in either env.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

MODELS = {
    "moge2": "Ruicheng/moge-2-vitl-normal",
    "da3": "depth-anything/DA3METRIC-LARGE",
}


def load_frames(capture: Path, count: int) -> list[np.ndarray]:
    paths = sorted((capture / "frames").glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {capture / 'frames'}")
    picks = np.linspace(0, len(paths) - 1, min(count, len(paths))).round().astype(int)
    return [cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB) for i in picks]


def pick_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def points_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - w / 2) * depth / fx, (v - h / 2) * depth / fy, depth], axis=-1)


def run_moge2(frames, device):
    import torch
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODELS["moge2"]).to(device).eval()
    depths, fxs, fys = [], [], []
    for rgb in frames:
        h, w = rgb.shape[:2]
        inp = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            # Accuracy first: full precision, highest token resolution.
            out = model.infer(inp, resolution_level=9, use_fp16=False)
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy() if "mask" in out else np.isfinite(depth)
        depth[~mask.astype(bool)] = np.nan
        k = out["intrinsics"].float().cpu().numpy()  # normalized by image size
        depths.append(depth)
        fxs.append(k[0, 0] * w)
        fys.append(k[1, 1] * h)
    return depths, float(np.median(fxs)), float(np.median(fys)), "moge2 estimate"


def run_da3(frames, device, focal_px: float, focal_source: str):
    # DA3's export helpers import pycolmap unconditionally; we never export to COLMAP,
    # and pycolmap's bundled libomp clashes with torch's on macOS.
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(MODELS["da3"]).to(device).eval()
    depths = []
    for rgb in frames:
        h, w = rgb.shape[:2]
        with torch.no_grad():
            pred = model.inference([rgb])
        raw = pred.depth[0].astype(np.float32)
        # Metric depth = focal * net_output / 300, focal in px at the network's resolution.
        focal_net = focal_px * raw.shape[1] / w
        depth = focal_net * raw / 300.0
        depths.append(cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR))
    return depths, focal_px, focal_px, focal_source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("capture", type=Path)
    ap.add_argument("--frames", type=int, default=5, help="frames to run (median-combined)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--hfov", type=float, help="DA3 only: camera HFOV (deg) if no moge2.npz focal is available")
    args = ap.parse_args()

    import torch

    device = pick_device(torch, args.device)
    frames = load_frames(args.capture, args.frames)
    h, w = frames[0].shape[:2]
    print(f"{args.model}: {len(frames)} frames {w}x{h} on {device}")

    t0 = time.perf_counter()
    if args.model == "moge2":
        depths, fx, fy, focal_source = run_moge2(frames, device)
    else:
        # DA3-Metric needs the focal length; reuse MoGe-2's estimate unless --hfov is given.
        moge = args.capture / "moge2.npz"
        if args.hfov:
            focal = (w / 2) / np.tan(np.radians(args.hfov) / 2)
            source = f"--hfov {args.hfov}"
        elif moge.exists():
            focal = float(np.load(moge)["fx"])
            source = "moge2 estimate"
        else:
            raise SystemExit("da3 needs a focal length: run moge2 first or pass --hfov")
        depths, fx, fy, focal_source = run_da3(frames, device, focal, source)
    elapsed = time.perf_counter() - t0

    depth = np.nanmedian(np.stack(depths), axis=0).astype(np.float32)
    out = args.capture / f"{args.model}.npz"
    np.savez_compressed(out, depth=depth, fx=fx, fy=fy, points=points_from_depth(depth, fx, fy),
                        focal_source=focal_source, model=MODELS[args.model], seconds=elapsed)
    hfov = np.degrees(2 * np.arctan((w / 2) / fx))
    print(f"saved {out}  ({elapsed:.1f}s, focal {fx:.0f}px = {hfov:.1f} deg HFOV from {focal_source}, "
          f"depth {np.nanpercentile(depth, 5):.2f}-{np.nanpercentile(depth, 95):.2f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
