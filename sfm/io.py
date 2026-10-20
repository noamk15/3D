"""Export: PLY point cloud and JSON scene for the Three.js viewer."""
from __future__ import annotations

import json

import numpy as np

from .geometry import camera_center


def write_ply(path, X, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(X, colors):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_viewer_json(path, K, poses, X, colors, width, height, gt_poses=None, meta=None):
    """poses: dict view -> (R, t). Camera frames are stored as centre + rotation
    (rows = camera axes in world) so the viewer can draw frustums."""
    cams = []
    for v in sorted(poses):
        R, t = poses[v]
        cams.append({"id": int(v), "C": camera_center(R, t).round(5).tolist(),
                     "R": np.asarray(R).round(6).tolist()})
    out = {
        "K": np.asarray(K).round(4).tolist(), "width": int(width), "height": int(height),
        "cameras": cams,
        "points": np.asarray(X, float).round(4).ravel().tolist(),
        "colors": np.asarray(colors, int).ravel().tolist(),
        "meta": meta or {},
    }
    if gt_poses is not None:
        out["gt_cameras"] = [{"id": int(v), "C": camera_center(R, t).round(5).tolist(),
                              "R": np.asarray(R).round(6).tolist()} for v, (R, t) in gt_poses.items()]
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
