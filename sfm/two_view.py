"""Two-view geometry: F (normalized 8-point), E, pose decomposition.

Convention: x2^T F x1 = 0, with x1 in image 1 and x2 in image 2
(homogeneous pixels). l2 = F x1 is the epipolar line of x1 in image 2.
"""
from __future__ import annotations

import numpy as np

from .geometry import to_h, skew, normalize_pixels
from .triangulation import triangulate_dlt


# ---------------------------------------------------------------------------
# Hartley normalization
# ---------------------------------------------------------------------------
def hartley_normalization(x: np.ndarray):
    """Similarity T moving the centroid to 0 and mean distance to sqrt(2).

    Returns (T, x_normalized) with x_normalized = T x (inhomogeneous).
    """
    mu = x.mean(axis=0)
    d = np.linalg.norm(x - mu, axis=1).mean()
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1.0]])
    xn = (x - mu) * s
    return T, xn


# ---------------------------------------------------------------------------
# Linear system A f = 0
# ---------------------------------------------------------------------------
def _design_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Row i encodes x2_i^T F x1_i = 0 as a_i . vec(F) = 0 (row-major vec)."""
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    one = np.ones_like(u1)
    return np.column_stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, one])


def enforce_rank2(F: np.ndarray) -> np.ndarray:
    """Closest rank-2 matrix in Frobenius norm (Eckart-Young): zero sigma_3."""
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    return U @ np.diag(S) @ Vt


def eight_point(x1: np.ndarray, x2: np.ndarray, normalize: bool = True,
                rank2: bool = True) -> np.ndarray:
    """Fundamental matrix from N >= 8 correspondences.

    1. (optional) Hartley-normalize each image: x_hat = T x.
    2. Solve min ||A f|| s.t. ||f|| = 1  -> last right singular vector of A.
    3. (optional) Project to rank 2 with an SVD.
    4. Undo normalization: F = T2^T F_hat T1.
    Output scaled to ||F||_F = 1.
    """
    assert len(x1) >= 8 and len(x1) == len(x2)
    if normalize:
        T1, x1n = hartley_normalization(x1)
        T2, x2n = hartley_normalization(x2)
    else:
        T1 = T2 = np.eye(3)
        x1n, x2n = x1, x2
    A = _design_matrix(x1n, x2n)
    _, _, Vt = np.linalg.svd(A, full_matrices=A.shape[0] < A.shape[1])  # thin SVD: never build the N x N U
    F = Vt[-1].reshape(3, 3)
    if rank2:
        F = enforce_rank2(F)
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def epipoles(F: np.ndarray):
    """e1 = right null vector of F (F e1 = 0), e2 = left (F^T e2 = 0)."""
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]
    return e1, e2


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
def algebraic_error(F, x1, x2):
    return np.einsum("ij,jk,ik->i", to_h(x2), F, to_h(x1))


def sampson_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Squared Sampson distance (pixels^2), first-order approx. of the
    geometric (reprojection) error:

        (x2^T F x1)^2 / ((F x1)_1^2 + (F x1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2)
    """
    h1, h2 = to_h(x1), to_h(x2)
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.einsum("ij,ij->i", h2, Fx1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / den


def symmetric_epipolar_distance(F, x1, x2):
    """Sum of squared point-to-epipolar-line distances in both images."""
    h1, h2 = to_h(x1), to_h(x2)
    l2 = h1 @ F.T
    l1 = h2 @ F
    r = np.einsum("ij,ij->i", h2, l2)
    return r**2 * (1 / (l2[:, 0]**2 + l2[:, 1]**2) + 1 / (l1[:, 0]**2 + l1[:, 1]**2))


# ---------------------------------------------------------------------------
# Essential matrix
# ---------------------------------------------------------------------------
def enforce_essential(E: np.ndarray) -> np.ndarray:
    """Project onto the essential manifold: singular values (1, 1, 0)."""
    U, S, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E


def essential_from_fundamental(F, K1, K2=None):
    K2 = K1 if K2 is None else K2
    return enforce_essential(K2.T @ F @ K1)


def essential_from_pose(R, t):
    return skew(t) @ R


def eight_point_essential(x1, x2, K1, K2=None):
    """E directly from normalized coordinates y = K^{-1} x (8-point on y)."""
    K2 = K1 if K2 is None else K2
    y1, y2 = normalize_pixels(K1, x1), normalize_pixels(K2, x2)
    E = eight_point(y1, y2, normalize=True, rank2=False)
    return enforce_essential(E)


def decompose_essential(E: np.ndarray):
    """The 4 candidate poses (R, t) with E ~ [t]x R, ||t|| = 1.

    E = U diag(1,1,0) V^T,  W = [[0,-1,0],[1,0,0],[0,0,1]]
    R in {U W V^T, U W^T V^T},  t in {+u3, -u3}.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    Ra = U @ W @ Vt
    Rb = U @ W.T @ Vt
    t = U[:, 2]
    return [(Ra, t), (Ra, -t), (Rb, t), (Rb, -t)]


def choose_pose_cheirality(E, x1, x2, K1, K2=None):
    """Pick the candidate for which most triangulated points lie in front
    of both cameras (positive depth). Returns (R, t, X, n_in_front_per_candidate).
    """
    K2 = K1 if K2 is None else K2
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best, counts = None, []
    for R, t in decompose_essential(E):
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
        X = triangulate_dlt(P1, P2, x1, x2)
        z1 = X[:, 2]
        z2 = X @ R[2] + t[2]
        n = int(np.sum((z1 > 0) & (z2 > 0)))
        counts.append(n)
        if best is None or n > best[3]:
            best = (R, t, X, n)
    return best[0], best[1], best[2], counts
