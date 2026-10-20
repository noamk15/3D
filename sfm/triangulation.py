"""Triangulation: linear DLT, then non-linear refinement of the
reprojection error (Gauss-Newton per point, vectorized).
"""
from __future__ import annotations

import numpy as np


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear triangulation of N points seen in two views.

    x ~ P X gives x (p3.X) - (p1.X) = 0 and y (p3.X) - (p2.X) = 0:
    two equations per view, A X = 0 with A 4x4, solved by SVD.
    """
    N = len(x1)
    A = np.empty((N, 4, 4))
    A[:, 0] = x1[:, [0]] * P1[2] - P1[0]
    A[:, 1] = x1[:, [1]] * P1[2] - P1[1]
    A[:, 2] = x2[:, [0]] * P2[2] - P2[0]
    A[:, 3] = x2[:, [1]] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    return Xh[:, :3] / Xh[:, 3:4]


def triangulate_nview_dlt(Ps: list, xs: np.ndarray) -> np.ndarray:
    """One point from n >= 2 views. Ps: list of 3x4, xs: (n, 2)."""
    rows = []
    for P, x in zip(Ps, xs):
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def _proj_and_jac(P: np.ndarray, X: np.ndarray):
    """Projection of points X (N,3) by P, and Jacobian d(u,v)/dX (N,2,3)."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    a = Xh @ P[0]
    b = Xh @ P[1]
    w = Xh @ P[2]
    u, v = a / w, b / w
    J = np.empty((len(X), 2, 3))
    J[:, 0] = (P[0, :3][None] * w[:, None] - P[2, :3][None] * a[:, None]) / w[:, None] ** 2
    J[:, 1] = (P[1, :3][None] * w[:, None] - P[2, :3][None] * b[:, None]) / w[:, None] ** 2
    return np.column_stack([u, v]), J


def refine_points(Ps: list, xs: list, X0: np.ndarray, iters: int = 10, masks=None) -> np.ndarray:
    """Gauss-Newton on sum_views ||x_v - pi(P_v X)||^2 for each point
    independently (3x3 normal equations per point, vectorized).

    Ps: list of V cameras; xs: list of V arrays (N,2); masks: optional list of
    V boolean arrays (N,) telling which views observe each point.
    """
    X = X0.copy()
    V = len(Ps)
    masks = masks if masks is not None else [np.ones(len(X), bool)] * V
    for _ in range(iters):
        H = np.zeros((len(X), 3, 3))
        g = np.zeros((len(X), 3))
        for P, x, m in zip(Ps, xs, masks):
            proj, J = _proj_and_jac(P, X)
            r = (x - proj) * m[:, None]
            J = J * m[:, None, None]
            H += np.einsum("nki,nkj->nij", J, J)
            g += np.einsum("nki,nk->ni", J, r)
        H += 1e-9 * np.eye(3)
        dX = np.linalg.solve(H, g[..., None])[..., 0]
        X = X + dX
        if np.max(np.abs(dX)) < 1e-12:
            break
    return X


def reprojection_errors(P: np.ndarray, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in pixels."""
    proj, _ = _proj_and_jac(P, X)
    return np.linalg.norm(proj - x, axis=1)


def triangulation_angle_deg(C1, C2, X):
    """Angle at X between rays to the two camera centres (small = unstable)."""
    a = C1[None] - X
    b = C2[None] - X
    c = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
