"""Block 7 figures: BA convergence on fountain-P11 and the Schur speed-up.

1. Start from the final fountain reconstruction, perturb every camera
   (~1 deg, ~2% of the scene) and every point, then run
   (a) our Levenberg-Marquardt + Schur, (b) scipy least_squares with the
   sparsity pattern. Plot RMS reprojection error vs. iteration / time.
2. Cost of ONE damped normal-equation solve: full (6n_c + 3n_p)^2 system vs.
   Schur reduced camera system (6n_c)^2, as the number of points grows.

    python -m experiments.ba_convergence
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from sfm.geometry import so3_exp
from sfm.incremental import Reconstruction
from sfm.ba import BAProblem, ba_lm_schur, ba_scipy, rms_reprojection
from sfm.synthetic import make_scene
from experiments.plotstyle import setup, line, plt, SERIES

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "docs", "figures")
OUT = os.path.join(HERE, "out")


def fountain_convergence():
    with open(os.path.join(OUT, "fountain_cache.pkl"), "rb") as f:
        C = pickle.load(f)
    rec = Reconstruction(K=C["K"], poses=dict(C["poses"]), X=C["X"].copy(), has_X=C["has_X"].copy())
    prob, views, tids = rec.ba_problem(C["tracks"], C["kps"])
    rng = np.random.default_rng(0)
    scene = np.linalg.norm(prob.X - prob.X.mean(0), axis=1).mean()
    Rs = [prob.Rs[0]] + [so3_exp(np.radians(1.0) * rng.normal(size=3) / np.sqrt(3)) @ R for R in prob.Rs[1:]]
    ts = [prob.ts[0]] + [t + 0.02 * scene * rng.normal(size=3) for t in prob.ts[1:]]
    X = prob.X + 0.01 * scene * rng.normal(size=prob.X.shape)
    pert = BAProblem(prob.K, Rs, ts, X, prob.cam_idx, prob.pt_idx, prob.uv)
    print(f"fountain BA: {len(Rs)} cams, {len(X)} points, {len(prob.uv)} obs, "
          f"{6 * len(Rs) + 3 * len(X)} unknowns; start rms {rms_reprojection(pert):.2f}px")
    a = ba_lm_schur(pert, iters=60, loss="l2", fix_scale=True)
    print(f"  LM+Schur: {a.iterations} it, {a.time_s:.1f}s, rms {a.rms_history[-1]:.4f}")
    b = ba_scipy(pert, loss="linear", max_nfev=60)
    print(f"  scipy sparse TRF: {b.iterations} nfev, {b.time_s:.1f}s, rms {b.rms_history[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    line(ax, np.arange(len(a.rms_history)), a.rms_history, 0, f"Our LM + Schur ({a.time_s:.1f} s)")
    ax.axhline(b.rms_history[-1], color=SERIES[1], lw=1.5, ls="--",
               label=f"scipy least_squares, sparse ({b.time_s:.1f} s), final")
    ax.set(title=f"BA convergence — fountain-P11 ({len(Rs)} cams, {len(X)} pts, {len(prob.uv)} obs)",
           xlabel="LM iteration", ylabel="RMS reprojection error (px)", yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ba_convergence_fountain.png"))
    plt.close(fig)
    return dict(n_cams=len(Rs), n_points=len(X), n_obs=len(prob.uv),
                start_rms=rms_reprojection(pert), lm_rms=a.rms_history, lm_time=a.time_s,
                lm_iters=a.iterations, scipy_rms=b.rms_history[-1], scipy_time=b.time_s,
                scipy_nfev=b.iterations)


def schur_scaling(n_cams=10):
    pts = [50, 100, 200, 400, 800, 1600, 3200]
    t_full, t_schur = [], []
    for n in pts:
        s = make_scene(n_points=n, n_cams=n_cams, arc_deg=80, seed=0)
        nu = 6 * n_cams + 3 * n
        # full system (dense): what you would do without exploiting sparsity
        if nu <= 6000:
            H = np.random.default_rng(0).normal(size=(nu, nu))
            H = H @ H.T + nu * np.eye(nu)
            g = np.ones(nu)
            np.linalg.solve(H, g)                      # warm-up
            t0 = time.perf_counter()
            for _ in range(3):
                np.linalg.solve(H, g)
            t_full.append((time.perf_counter() - t0) / 3)
        else:
            t_full.append(np.nan)
        # Schur: one LM iteration of our solver (linearize + reduce + solve)
        ci = np.repeat(np.arange(n_cams), n)
        pi_ = np.tile(np.arange(n), n_cams)
        prob = BAProblem(s.K, s.Rs, s.ts, s.X + 0.01, ci, pi_, np.concatenate(s.obs))
        ts_ = []
        for _ in range(3):
            t0 = time.perf_counter()
            ba_lm_schur(prob, iters=1)
            ts_.append(time.perf_counter() - t0)
        t_schur.append(min(ts_))
        print(f"n_points {n}: unknowns {nu}, dense solve {t_full[-1]:.4f}s, Schur LM iteration {t_schur[-1]:.4f}s")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.array(pts)
    line(ax, x, np.array(t_full), 1, "Dense solve of the full normal equations (solve only)")
    line(ax, x, np.array(t_schur), 0, "Our full LM iteration with Schur complement")
    ax.set(title=f"Cost per iteration vs. number of points ({n_cams} cameras)",
           xlabel="number of 3D points", ylabel="seconds", xscale="log", yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "schur_scaling.png"))
    plt.close(fig)
    return dict(points=pts, t_full=t_full, t_schur=t_schur)


if __name__ == "__main__":
    setup()
    os.makedirs(FIG, exist_ok=True)
    out = dict(fountain=fountain_convergence(), scaling=schur_scaling())
    with open(os.path.join(OUT, "ba_convergence.json"), "w") as f:
        json.dump(out, f)
