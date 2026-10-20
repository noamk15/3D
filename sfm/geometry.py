"""Basic projective geometry and SO(3) tools.

Conventions used everywhere in this project
-------------------------------------------
* A camera is (K, R, t) with projection P = K [R | t].
  A world point X maps to camera coordinates X_c = R X + t, and the camera
  centre is C = -R^T t.
* 2D points are arrays of shape (N, 2) in pixels; 3D points are (N, 3).
* Homogeneous coordinates are only used internally.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Homogeneous coordinates
# ---------------------------------------------------------------------------
def to_h(x: np.ndarray) -> np.ndarray:
    """(N, d) -> (N, d+1) by appending a column of ones."""
    x = np.asarray(x, dtype=float)
    return np.hstack([x, np.ones((x.shape[0], 1))])


def from_h(xh: np.ndarray) -> np.ndarray:
    """(N, d+1) -> (N, d) by dividing by the last coordinate."""
    xh = np.asarray(xh, dtype=float)
    return xh[:, :-1] / xh[:, -1:]


def skew(v: np.ndarray) -> np.ndarray:
    """[v]_x such that [v]_x w = v x w."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D points X (N,3) to pixels (N,2)."""
    Xc = X @ R.T + t.reshape(1, 3)
    x = Xc @ K.T
    return x[:, :2] / x[:, 2:3]


def depths(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Depth (z in camera frame) of each 3D point."""
    return X @ R[2] + t[2]


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def normalize_pixels(K: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pixels -> normalized image coordinates K^{-1} x (still (N,2))."""
    return from_h(to_h(x) @ np.linalg.inv(K).T)


# ---------------------------------------------------------------------------
# SO(3): exponential / logarithm maps (Rodrigues)
# ---------------------------------------------------------------------------
def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector w in so(3) (axis * angle) -> R in SO(3).

    R = I + sin(th)/th [w]x + (1 - cos(th))/th^2 [w]x^2   (Rodrigues).
    A Taylor expansion is used for small angles to stay numerically exact.
    """
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    W = skew(w)
    if th < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * W @ W


def so3_log(R: np.ndarray) -> np.ndarray:
    """R in SO(3) -> rotation vector w with |w| in [0, pi]."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:
        # Near pi: axis is the eigenvector of R with eigenvalue 1.
        M = (R + np.eye(3)) / 2
        axis = M[:, np.argmax(np.diag(M))]
        axis = axis / np.linalg.norm(axis)
        return axis * th
    return th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    """Closest rotation in Frobenius norm (SVD, with det fix)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    c = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def direction_error_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Angle between two directions (translation known up to scale), degrees."""
    c = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation (via a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """Best similarity (s, R, t) with dst ~ s R src + t (least squares).

    Used to compare a reconstruction (defined up to a similarity) with
    ground truth.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    var_s = (S**2).sum() / len(src)
    s = np.trace(np.diag(sig) @ E) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------
# Projection with analytic Jacobians (shared by PnP refinement and BA)
# ---------------------------------------------------------------------------
def project_with_jacobians(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray):
    """u = pi(K (R X + t)).

    Pose update is parametrized in so(3) on the LEFT:
        R <- exp([d_w]x) R,  t <- t + d_t
    so that d(X_c)/d(d_w) = -[R X]x  and  d(X_c)/d(d_t) = I.

    Returns uv (N,2), J_pose (N,2,6) wrt (d_w, d_t), J_X (N,2,3).
    """
    RX = X @ R.T
    Xc = RX + t.reshape(1, 3)
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    fx, fy, s = K[0, 0], K[1, 1], K[0, 1]
    uv = np.column_stack([fx * x / z + s * y / z + K[0, 2], fy * y / z + K[1, 2]])
    # d uv / d Xc   (N,2,3)
    Jp = np.zeros((len(X), 2, 3))
    Jp[:, 0, 0] = fx / z
    Jp[:, 0, 1] = s / z
    Jp[:, 0, 2] = -(fx * x + s * y) / z**2
    Jp[:, 1, 1] = fy / z
    Jp[:, 1, 2] = -fy * y / z**2
    # d Xc / d d_w = -[RX]x  (N,3,3)
    S = np.zeros((len(X), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = RX[:, 2], -RX[:, 1]
    S[:, 1, 0], S[:, 1, 2] = -RX[:, 2], RX[:, 0]
    S[:, 2, 0], S[:, 2, 1] = RX[:, 1], -RX[:, 0]
    J_pose = np.concatenate([Jp @ S, Jp], axis=2)
    J_X = Jp @ R
    return uv, J_pose, J_X
