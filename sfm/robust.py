"""Robust estimation: RANSAC / MSAC / LO-RANSAC, robust losses (IRLS).

RANSAC iteration count: to draw, with probability p, at least one sample
of s inliers when the inlier ratio is w:
        N = log(1 - p) / log(1 - w^s)
w is unknown, so it is re-estimated from the best consensus so far
(adaptive termination).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .two_view import eight_point, sampson_error


def ransac_iterations(p: float, w: float, s: int, cap: int = 100_000) -> int:
    if w <= 0:
        return cap
    if w >= 1:
        return 1
    denom = np.log(1 - w**s)
    if denom == 0:
        return cap
    return int(min(cap, np.ceil(np.log(1 - p) / denom)))


@dataclass
class RansacResult:
    model: np.ndarray
    inliers: np.ndarray
    iterations: int
    time_s: float


def ransac(data_n: int, fit, residuals_sq, sample_size: int, threshold: float,
           method: str = "ransac", confidence: float = 0.999,
           max_iters: int = 20_000, min_iters: int = 50, seed: int = 0,
           lo_iters: int = 5) -> RansacResult:
    """Generic hypothesize-and-verify loop.

    fit(idx) -> model or None ;  residuals_sq(model) -> (data_n,) squared errors.
    method:
      'ransac' : score = number of points with e^2 < T^2
      'msac'   : score = - sum min(e^2, T^2)   (truncated quadratic cost:
                 among models with the same inlier count, prefers the tighter one)
      'lo'     : MSAC + local optimization: when a new best model appears,
                 re-fit on its inliers a few times (LO-RANSAC, Chum et al. 2003)
    """
    rng = np.random.default_rng(seed)
    T2 = threshold**2
    best_score, best_model, best_inl = -np.inf, None, np.zeros(data_n, bool)
    n_needed, it = max_iters, 0
    t0 = time.perf_counter()

    def score_of(e2):
        if method == "ransac":
            return float(np.sum(e2 < T2))
        return -float(np.sum(np.minimum(e2, T2)))

    while it < min(n_needed, max_iters) or it < min_iters:
        it += 1
        idx = rng.choice(data_n, sample_size, replace=False)
        model = fit(idx)
        if model is None:
            continue
        e2 = residuals_sq(model)
        sc = score_of(e2)
        if sc > best_score:
            best_score, best_model, best_inl = sc, model, e2 < T2
            if method == "lo" and best_inl.sum() > sample_size:
                for _ in range(lo_iters):
                    m2 = fit(np.flatnonzero(best_inl))
                    if m2 is None:
                        break
                    e2b = residuals_sq(m2)
                    sc2 = score_of(e2b)
                    if sc2 <= best_score:
                        break
                    best_score, best_model, best_inl = sc2, m2, e2b < T2
            w = best_inl.mean()
            n_needed = ransac_iterations(confidence, w, sample_size, max_iters)
        if it >= max_iters:
            break
    return RansacResult(best_model, best_inl, it, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Fundamental matrix
# ---------------------------------------------------------------------------
def ransac_fundamental(x1, x2, threshold: float = 1.0, method: str = "ransac",
                       confidence: float = 0.999, seed: int = 0, refit: bool = True,
                       max_iters: int = 20_000) -> RansacResult:
    """F with RANSAC, 8-point minimal(ish) samples, Sampson residual.
    The final model is re-estimated on all inliers (least squares)."""
    def fit(idx):
        try:
            return eight_point(x1[idx], x2[idx])
        except np.linalg.LinAlgError:
            return None

    def res(F):
        return sampson_error(F, x1, x2)

    r = ransac(len(x1), fit, res, 8, threshold, method, confidence,
               max_iters=max_iters, seed=seed)
    if refit and r.inliers.sum() >= 8:
        # The minimal-sample model is noisy (fit to 8 points only): re-estimate
        # by least squares on ALL inliers, re-classify, repeat once.
        inl = r.inliers
        for _ in range(2):
            F = eight_point(x1[inl], x2[inl])
            new_inl = sampson_error(F, x1, x2) < threshold**2
            if new_inl.sum() < 8:
                break
            inl = new_inl
        r = RansacResult(F, inl, r.iterations, r.time_s)
    return r


# ---------------------------------------------------------------------------
# Robust losses: rho(r^2) as in scipy, and IRLS weights w(r) = rho'(r)/r
# ---------------------------------------------------------------------------
def robust_weights(r: np.ndarray, loss: str = "l2", scale: float = 1.0) -> np.ndarray:
    """IRLS weight for residual magnitudes r (one weight per residual).

    l2     : w = 1                              (outliers have unbounded influence)
    huber  : w = 1 if |r|<=c else c/|r|         (influence bounded, convex)
    cauchy : w = 1 / (1 + (r/c)^2)              (influence -> 0, redescending,
                                                 non-convex)
    """
    a = np.abs(r) / scale
    if loss == "l2":
        return np.ones_like(a)
    if loss == "huber":
        return np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12))
    if loss == "cauchy":
        return 1.0 / (1.0 + a**2)
    raise ValueError(loss)


def robust_cost(r: np.ndarray, loss: str = "l2", scale: float = 1.0) -> float:
    """sum rho(r): l2 -> r^2/2 ; huber ; cauchy (c^2/2) log(1+(r/c)^2)."""
    a = np.abs(r)
    c = scale
    if loss == "l2":
        return float(0.5 * np.sum(a**2))
    if loss == "huber":
        return float(np.sum(np.where(a <= c, 0.5 * a**2, c * a - 0.5 * c**2)))
    if loss == "cauchy":
        return float(np.sum(0.5 * c**2 * np.log1p((a / c) ** 2)))
    raise ValueError(loss)
