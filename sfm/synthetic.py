"""Synthetic scenes with exact ground truth, to test every block.

A scene = 3D points + n cameras placed on an arc looking at the origin.
Knobs: pixel noise sigma, outlier ratio, baseline (arc angle), planar scene,
pure rotation. Everything is seeded, hence reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import project, depths


def default_K(f: float = 800.0, w: int = 1024, h: int = 768) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def look_at(C: np.ndarray, target: np.ndarray, up=np.array([0, -1.0, 0])):
    """Rotation R, translation t of a camera at C looking at target."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])          # rows = camera axes in world frame
    t = -R @ C
    return R, t


@dataclass
class Scene:
    K: np.ndarray
    Rs: list
    ts: list
    X: np.ndarray                      # (N,3) ground-truth points
    obs: list = field(default_factory=list)   # obs[i] = (N,2) pixels in camera i
    width: int = 1024
    height: int = 768


def make_scene(
    n_points: int = 200,
    n_cams: int = 2,
    arc_deg: float = 30.0,
    radius: float = 6.0,
    noise_px: float = 0.0,
    planar: bool = False,
    pure_rotation: bool = False,
    seed: int = 0,
    K: np.ndarray | None = None,
) -> Scene:
    rng = np.random.default_rng(seed)
    K = default_K() if K is None else K
    if planar:
        X = np.column_stack([rng.uniform(-1.5, 1.5, n_points),
                             rng.uniform(-1.5, 1.5, n_points),
                             np.zeros(n_points)])
    else:
        X = rng.uniform(-1.5, 1.5, size=(n_points, 3))

    Rs, ts = [], []
    angles = np.radians(np.linspace(-arc_deg / 2, arc_deg / 2, n_cams))
    for k, a in enumerate(angles):
        if pure_rotation:
            C = np.array([0.0, 0.0, -radius])
            target = np.array([radius * np.sin(a), 0.3 * np.sin(2 * a), 0.0])
        else:
            C = np.array([radius * np.sin(a), 0.5 * rng.uniform(-1, 1), -radius * np.cos(a)])
            target = 0.2 * rng.normal(size=3)
        R, t = look_at(C, target)
        Rs.append(R)
        ts.append(t)

    obs = []
    for R, t in zip(Rs, ts):
        assert np.all(depths(R, t, X) > 0), "point behind a camera"
        x = project(K, R, t, X) + noise_px * rng.normal(size=(n_points, 2))
        obs.append(x)
    return Scene(K=K, Rs=Rs, ts=ts, X=X, obs=obs)


def relative_pose(R1, t1, R2, t2):
    """Pose of camera 2 relative to camera 1 (camera 1 = [I|0])."""
    R = R2 @ R1.T
    t = t2 - R @ t1
    return R, t


def add_outliers(x: np.ndarray, ratio: float, width: int = 1024, height: int = 768,
                 seed: int = 0):
    """Replace a fraction `ratio` of the points by uniform random pixels.

    Returns (corrupted copy, boolean inlier mask).
    """
    rng = np.random.default_rng(seed)
    x = x.copy()
    n = len(x)
    k = int(round(ratio * n))
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = np.column_stack([rng.uniform(0, width, k), rng.uniform(0, height, k)])
    inl = np.ones(n, dtype=bool)
    inl[idx] = False
    return x, inl
