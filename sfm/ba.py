"""Bundle adjustment.

Minimize over cameras (R_c, t_c) and points X_p
        sum_obs rho( || u_obs - pi(K (R_c X_p + t_c)) || )
Two implementations:
  * `ba_lm_schur` : Levenberg-Marquardt written by hand. Rotations are updated
    in so(3) (R <- exp([dw]x) R), robust losses via IRLS weights, and the
    normal equations are solved with the Schur complement (reduced camera
    system), exploiting the block-sparse structure of the Jacobian.
  * `ba_scipy`    : scipy.optimize.least_squares (trust region reflective)
    with an explicit Jacobian sparsity pattern; rotations as rotation vectors.

Gauge: the cost is invariant to a global similarity (7 dof). We fix camera 0
(6 dof) and one translation coordinate of another camera (scale, 1 dof).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .geometry import project_with_jacobians, so3_exp, so3_log
from .robust import robust_weights, robust_cost


@dataclass
class BAProblem:
    K: np.ndarray
    Rs: list                 # n_c rotations
    ts: list                 # n_c translations
    X: np.ndarray            # (n_p, 3)
    cam_idx: np.ndarray      # (M,)
    pt_idx: np.ndarray       # (M,)
    uv: np.ndarray           # (M, 2)

    def copy(self):
        return BAProblem(self.K, [R.copy() for R in self.Rs], [t.copy() for t in self.ts],
                         self.X.copy(), self.cam_idx, self.pt_idx, self.uv)


@dataclass
class BAResult:
    problem: BAProblem
    cost_history: list = field(default_factory=list)
    rms_history: list = field(default_factory=list)
    time_s: float = 0.0
    iterations: int = 0


def residuals(prob: BAProblem) -> np.ndarray:
    """(M,2) residuals u_obs - pi(...)."""
    r = np.empty_like(prob.uv)
    for c in range(len(prob.Rs)):
        m = prob.cam_idx == c
        if not m.any():
            continue
        R, t = prob.Rs[c], prob.ts[c]
        Xc = prob.X[prob.pt_idx[m]] @ R.T + t
        x = Xc @ prob.K.T
        r[m] = prob.uv[m] - x[:, :2] / x[:, 2:3]
    return r


def rms_reprojection(prob: BAProblem) -> float:
    return float(np.sqrt(np.mean(np.sum(residuals(prob) ** 2, axis=1))))


def _scale_gauge(prob: BAProblem, fixed_cams):
    """Pick (camera, coordinate) whose translation entry is frozen to fix scale."""
    free = [c for c in range(len(prob.Rs)) if c not in fixed_cams]
    if not free:
        return None
    c = free[0]
    return c, int(np.argmax(np.abs(prob.ts[c])))


# ---------------------------------------------------------------------------
# Hand-written Levenberg-Marquardt with Schur complement
# ---------------------------------------------------------------------------
def ba_lm_schur(prob: BAProblem, iters: int = 50, loss: str = "l2", loss_scale: float = 2.0,
                fixed_cams=(0,), fix_scale: bool = True, lam0: float = 1e-3,
                tol: float = 1e-10, verbose: bool = False) -> BAResult:
    prob = prob.copy()
    nc, n_p = len(prob.Rs), len(prob.X)
    ci, pi_ = prob.cam_idx, prob.pt_idx
    fixed = set(fixed_cams)
    scale_fix = _scale_gauge(prob, fixed) if fix_scale else None
    free_mask = np.ones(6 * nc, bool)
    for c in fixed:
        free_mask[6 * c:6 * c + 6] = False
    if scale_fix is not None:
        free_mask[6 * scale_fix[0] + 3 + scale_fix[1]] = False

    def cost(p):
        r = np.linalg.norm(residuals(p), axis=1)
        return robust_cost(r, loss, loss_scale)

    lam = lam0
    res = BAResult(prob)
    cur = cost(prob)
    res.cost_history.append(cur)
    res.rms_history.append(rms_reprojection(prob))
    t0 = time.perf_counter()
    for it in range(iters):
        # ---- linearize: per-observation Jacobian blocks -----------------
        M = len(ci)
        Jc = np.zeros((M, 2, 6))
        Jx = np.zeros((M, 2, 3))
        r = np.zeros((M, 2))
        for c in range(nc):
            m = ci == c
            if not m.any():
                continue
            uv, Jp, JX = project_with_jacobians(prob.K, prob.Rs[c], prob.ts[c], prob.X[pi_[m]])
            Jc[m], Jx[m], r[m] = Jp, JX, prob.uv[m] - uv
        w = robust_weights(np.linalg.norm(r, axis=1), loss, loss_scale)   # IRLS
        # ---- normal-equation blocks  H = [[U, W],[W^T, V]] ------------------
        U = np.zeros((nc, 6, 6))
        np.add.at(U, ci, np.einsum("m,mki,mkj->mij", w, Jc, Jc))
        V = np.zeros((n_p, 3, 3))
        np.add.at(V, pi_, np.einsum("m,mki,mkj->mij", w, Jx, Jx))
        Wm = np.einsum("m,mki,mkj->mij", w, Jc, Jx)                 # (M,6,3)
        gc = np.zeros((nc, 6))
        np.add.at(gc, ci, np.einsum("m,mki,mk->mi", w, Jc, r))
        gp = np.zeros((n_p, 3))
        np.add.at(gp, pi_, np.einsum("m,mki,mk->mi", w, Jx, r))

        while True:
            # ---- damping (Marquardt: scale by the diagonal) ---------------
            Ud = U.copy()
            idx = np.arange(6)
            Ud[:, idx, idx] += lam * (U[:, idx, idx] + 1e-9)
            Vd = V.copy()
            idx3 = np.arange(3)
            Vd[:, idx3, idx3] += lam * (V[:, idx3, idx3] + 1e-9)
            Vinv = np.linalg.inv(Vd)                                   # 3x3 per point
            # ---- Schur complement: S = U - W V^-1 W^T --------------------
            # per point, dense (6 nc x 3) block W_p  (cameras are few)
            Wp = np.zeros((n_p, 6 * nc, 3))
            for c in range(nc):
                m = ci == c
                Wp[pi_[m], 6 * c:6 * c + 6, :] = Wm[m]
            Y = Wp @ Vinv                                              # (n_p, 6nc, 3)
            S = -(Y.transpose(1, 0, 2).reshape(6 * nc, -1)
                  @ Wp.transpose(1, 0, 2).reshape(6 * nc, -1).T)
            for c in range(nc):
                S[6 * c:6 * c + 6, 6 * c:6 * c + 6] += Ud[c]
            b = gc.reshape(-1) - np.einsum("pai,pi->a", Y, gp)
            # gauge: remove fixed parameters
            fm = free_mask
            dc = np.zeros(6 * nc)
            dc[fm] = np.linalg.solve(S[np.ix_(fm, fm)], b[fm])
            # back-substitution for the points
            dp = np.einsum("pij,pj->pi", Vinv, gp - np.einsum("pai,a->pi", Wp, dc))
            # ---- try the step ---------------------------------------------
            cand = prob.copy()
            for c in range(nc):
                d = dc[6 * c:6 * c + 6]
                cand.Rs[c] = so3_exp(d[:3]) @ prob.Rs[c]
                cand.ts[c] = prob.ts[c] + d[3:]
            cand.X = prob.X + dp
            new = cost(cand)
            if new < cur:
                rel = (cur - new) / max(cur, 1e-300)
                prob, cur = cand, new
                lam = max(lam / 3, 1e-12)
                break
            lam *= 5
            if lam > 1e12:
                rel = 0.0
                break
        res.cost_history.append(cur)
        res.rms_history.append(rms_reprojection(prob))
        res.iterations = it + 1
        if verbose:
            print(f"it {it:3d} cost {cur:.6e} rms {res.rms_history[-1]:.4f} lam {lam:.1e}")
        if rel < tol or lam > 1e12:
            break
    res.time_s = time.perf_counter() - t0
    res.problem = prob
    return res


def solve_normal_equations_dense(prob: BAProblem, lam: float = 1e-3, fixed_cams=(0,)):
    """Reference: one damped GN step solving the FULL system (no Schur).
    Used only in tests to check that the Schur elimination gives the same step."""
    nc, n_p = len(prob.Rs), len(prob.X)
    n = 6 * nc + 3 * n_p
    J = np.zeros((2 * len(prob.uv), n))
    r = np.zeros(2 * len(prob.uv))
    for c in range(nc):
        m = np.flatnonzero(prob.cam_idx == c)
        uv, Jp, JX = project_with_jacobians(prob.K, prob.Rs[c], prob.ts[c], prob.X[prob.pt_idx[m]])
        for k, o in enumerate(m):
            J[2 * o:2 * o + 2, 6 * c:6 * c + 6] = Jp[k]
            p = prob.pt_idx[o]
            J[2 * o:2 * o + 2, 6 * nc + 3 * p:6 * nc + 3 * p + 3] = JX[k]
            r[2 * o:2 * o + 2] = prob.uv[o] - uv[k]
    H = J.T @ J
    H[np.diag_indices(n)] += lam * (np.diag(H) + 1e-9)
    g = J.T @ r
    free = np.ones(n, bool)
    for c in fixed_cams:
        free[6 * c:6 * c + 6] = False
    d = np.zeros(n)
    d[free] = np.linalg.solve(H[np.ix_(free, free)], g[free])
    return d, J


# ---------------------------------------------------------------------------
# scipy version (sparse Jacobian)
# ---------------------------------------------------------------------------
def ba_scipy(prob: BAProblem, loss: str = "linear", loss_scale: float = 2.0,
             fixed_cams=(0,), use_sparsity: bool = True, max_nfev: int = 100,
             verbose: int = 0) -> BAResult:
    prob = prob.copy()
    nc, n_p = len(prob.Rs), len(prob.X)
    free_cams = [c for c in range(nc) if c not in fixed_cams]
    pos = {c: k for k, c in enumerate(free_cams)}
    nfc = len(free_cams)

    def unpack(x):
        Rs = list(prob.Rs)
        ts = list(prob.ts)
        for c in free_cams:
            k = pos[c]
            Rs[c] = so3_exp(x[6 * k:6 * k + 3])
            ts[c] = x[6 * k + 3:6 * k + 6]
        X = x[6 * nfc:].reshape(n_p, 3)
        return Rs, ts, X

    def fun(x):
        Rs, ts, X = unpack(x)
        Rall = np.stack(Rs)[prob.cam_idx]
        tall = np.stack(ts)[prob.cam_idx]
        Xc = np.einsum("mij,mj->mi", Rall, X[prob.pt_idx]) + tall
        u = Xc @ prob.K.T
        return (u[:, :2] / u[:, 2:3] - prob.uv).ravel()

    x0 = np.concatenate([np.concatenate([so3_log(prob.Rs[c]), prob.ts[c]]) for c in free_cams]
                        + [prob.X.ravel()]) if nfc else prob.X.ravel()
    sparsity = None
    if use_sparsity:
        M = len(prob.uv)
        A = lil_matrix((2 * M, x0.size), dtype=int)
        rows = np.arange(M)
        for k in range(6):
            for c in free_cams:
                m = rows[prob.cam_idx == c]
                A[2 * m, 6 * pos[c] + k] = 1
                A[2 * m + 1, 6 * pos[c] + k] = 1
        for k in range(3):
            A[2 * rows, 6 * nfc + 3 * prob.pt_idx + k] = 1
            A[2 * rows + 1, 6 * nfc + 3 * prob.pt_idx + k] = 1
        sparsity = A
    t0 = time.perf_counter()
    sol = least_squares(fun, x0, jac_sparsity=sparsity, method="trf", loss=loss,
                        f_scale=loss_scale, x_scale="jac", max_nfev=max_nfev,
                        verbose=verbose)
    dt = time.perf_counter() - t0
    Rs, ts, X = unpack(sol.x)
    prob.Rs, prob.ts, prob.X = Rs, ts, X
    out = BAResult(prob, time_s=dt, iterations=sol.nfev)
    out.cost_history = [float(sol.cost)]
    out.rms_history = [rms_reprojection(prob)]
    return out
