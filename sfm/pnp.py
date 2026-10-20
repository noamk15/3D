"""Camera resectioning (PnP) with known K: linear DLT + RANSAC +
Gauss-Newton refinement of the reprojection error on SE(3).
"""
from __future__ import annotations

import numpy as np

from .geometry import (normalize_pixels, project, project_with_jacobians,
                       so3_exp, project_to_SO3)
from .robust import ransac


def pnp_dlt(X: np.ndarray, x: np.ndarray, K: np.ndarray):
    """Pose from n >= 6 2D-3D correspondences.

    With y = K^{-1} x we have y ~ [R | t] X: 2 linear equations per point in
    the 12 entries of M = [R|t]. Solve A m = 0 by SVD (after normalizing the
    3D points), then project the left 3x3 block onto SO(3) and fix the scale
    and sign (det R = +1).
    """
    y = normalize_pixels(K, x)
    mu = X.mean(0)
    sc = np.sqrt(3) / np.linalg.norm(X - mu, axis=1).mean()
    T = np.diag([sc, sc, sc, 1.0])
    T[:3, 3] = -sc * mu
    Xn = (X - mu) * sc
    Xh = np.hstack([Xn, np.ones((len(X), 1))])
    n = len(X)
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -y[:, [0]] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -y[:, [1]] * Xh
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4) @ T
    if np.linalg.det(M[:, :3]) < 0:
        M = -M
    U, S, Vt3 = np.linalg.svd(M[:, :3])
    R = U @ Vt3
    t = M[:, 3] / S.mean()
    return project_to_SO3(R), t


def refine_pose(K, R, t, X, x, iters: int = 15, weights=None):
    """Gauss-Newton on sum ||x - pi(K(RX+t))||^2 with left so(3) updates."""
    for _ in range(iters):
        uv, J, _ = project_with_jacobians(K, R, t, X)
        r = (x - uv)
        w = np.ones(len(X)) if weights is None else weights
        H = np.einsum("n,nki,nkj->ij", w, J, J) + 1e-9 * np.eye(6)
        g = np.einsum("n,nki,nk->i", w, J, r)
        d = np.linalg.solve(H, g)
        R = so3_exp(d[:3]) @ R
        t = t + d[3:]
        if np.abs(d).max() < 1e-12:
            break
    return R, t


def pnp_ransac(X, x, K, threshold: float = 2.0, method: str = "msac", seed: int = 0,
               confidence: float = 0.999, max_iters: int = 5000):
    """Robust PnP. Minimal-ish DLT sample of 6 points; residual = squared
    reprojection error (with a cheirality check). Returns R, t, inliers."""
    def fit(idx):
        try:
            return pnp_dlt(X[idx], x[idx], K)
        except np.linalg.LinAlgError:
            return None

    def res(model):
        R, t = model
        z = X @ R[2] + t[2]
        e2 = np.sum((project(K, R, t, X) - x) ** 2, axis=1)
        e2[z <= 0] = np.inf
        return e2

    r = ransac(len(X), fit, res, 6, threshold, method, confidence,
               max_iters=max_iters, seed=seed)
    if r.model is None or r.inliers.sum() < 6:
        return None, None, np.zeros(len(X), bool)
    R, t = pnp_dlt(X[r.inliers], x[r.inliers], K)
    R, t = refine_pose(K, R, t, X[r.inliers], x[r.inliers])
    inl = res((R, t)) < threshold**2
    return R, t, inl
