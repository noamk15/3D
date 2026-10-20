"""Blocks 1-3 figures: why normalize, degenerate configurations.

1. Normalized vs. unnormalized 8-point: error vs. pixel noise, and the
   condition number of A^T A.
2. Degeneracies, seen in the singular values of the 8-point design matrix A:
   general scene (1-dim null space) vs. planar scene (3-dim null space:
   F not unique) vs. pure rotation (E = 0: translation unobservable).
3. Triangulation error vs. baseline (depth uncertainty ~ 1/baseline).

    python -m experiments.conditioning
"""
from __future__ import annotations

import os

import numpy as np

from sfm.geometry import projection_matrix, rotation_error_deg
from sfm.synthetic import make_scene, relative_pose
from sfm.two_view import (eight_point, sampson_error, _design_matrix, hartley_normalization,
                          essential_from_fundamental, choose_pose_cheirality)
from sfm.triangulation import triangulate_dlt
from experiments.plotstyle import setup, line, plt

FIG = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


def gt_sampson_rms(F, s):
    """RMS geometric (Sampson) error of F evaluated on the NOISE-FREE points."""
    x1 = s.obs_clean[0]
    x2 = s.obs_clean[1]
    return np.sqrt(sampson_error(F, x1, x2).mean())


def normalization_study():
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    res = {"norm": [], "raw": [], "cond_norm": [], "cond_raw": [], "rot_norm": [], "rot_raw": []}
    for sg in sigmas:
        en, er, cn, cr, rn, rr = [], [], [], [], [], []
        for trial in range(40):
            s = make_scene(n_points=100, noise_px=sg, seed=trial)
            sc = make_scene(n_points=100, noise_px=0.0, seed=trial)
            s.obs_clean = sc.obs
            Rgt, _ = relative_pose(s.Rs[0], s.ts[0], s.Rs[1], s.ts[1])
            for norm, e, c, rlist in ((True, en, cn, rn), (False, er, cr, rr)):
                F = eight_point(s.obs[0], s.obs[1], normalize=norm)
                e.append(gt_sampson_rms(F, s))
                if norm:
                    A = _design_matrix(hartley_normalization(s.obs[0])[1], hartley_normalization(s.obs[1])[1])
                else:
                    A = _design_matrix(s.obs[0], s.obs[1])
                sv = np.linalg.svd(A, compute_uv=False)
                c.append(sv[0] / sv[-2])        # condition of the solution subspace
                E = essential_from_fundamental(F, s.K)
                R, _, _, _ = choose_pose_cheirality(E, s.obs[0], s.obs[1], s.K)
                rlist.append(rotation_error_deg(R, Rgt))
        res["norm"].append(en); res["raw"].append(er)
        res["cond_norm"].append(cn); res["cond_raw"].append(cr)
        res["rot_norm"].append(rn); res["rot_raw"].append(rr)
        print(f"sigma {sg}: normalized {np.median(en):.3g}px rot {np.median(rn):.3g} | raw {np.median(er):.3g}px rot {np.median(rr):.3g} "
              f"| cond {np.median(cn):.2e} vs {np.median(cr):.2e}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.array(sigmas)
    for k, (key, rkey, lab) in enumerate((("norm", "rot_norm", "Normalized (Hartley)"),
                                          ("raw", "rot_raw", "Raw pixels"))):
        a = np.array(res[key]); r = np.array(res[rkey])
        line(axes[0], x[1:], np.median(a, 1)[1:], k, lab,
             (np.percentile(a, 25, 1)[1:], np.percentile(a, 75, 1)[1:]))
        line(axes[1], x[1:], np.median(r, 1)[1:], k, lab,
             (np.percentile(r, 25, 1)[1:], np.percentile(r, 75, 1)[1:]))
    axes[0].set(title="Geometric error of F (on noise-free points)", xlabel="pixel noise σ",
                ylabel="RMS Sampson distance (px)", xscale="log", yscale="log")
    axes[1].set(title="Rotation error from E = KᵀFK", xlabel="pixel noise σ",
                ylabel="degrees", xscale="log", yscale="log")
    axes[0].legend(fontsize=8)
    cn, cr = np.median(res["cond_norm"]), np.median(res["cond_raw"])
    fig.suptitle(f"Normalized vs. unnormalized 8-point (100 matches, 40 trials) — "
                 f"median σ₁/σ₈ of A: {cn:.0f} vs. {cr:.1e}",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "normalization.png"))
    plt.close(fig)
    return cn, cr


def degeneracy_study():
    cases = [("General 3D scene", dict()), ("Planar scene", dict(planar=True)),
             ("Pure rotation (t = 0)", dict(pure_rotation=True))]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, (name, kw) in enumerate(cases):
        s = make_scene(n_points=100, noise_px=0.0, seed=3, **kw)
        A = _design_matrix(hartley_normalization(s.obs[0])[1], hartley_normalization(s.obs[1])[1])
        sv = np.linalg.svd(A, compute_uv=False)
        line(ax, np.arange(1, 10), sv / sv[0], k, name)
        print(name, np.round(sv / sv[0], 12))
    ax.set(title="Singular values of the 8-point matrix A (noise-free)", xlabel="index i",
           ylabel="σᵢ / σ₁", yscale="log")
    ax.set_xticks(range(1, 10))
    ax.legend(fontsize=8)
    ax.annotate("1 zero → F unique", (9, 1e-15), xytext=(5.6, 1e-13), fontsize=8, color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "degeneracies.png"))
    plt.close(fig)


def baseline_study():
    arcs = [0.25, 0.5, 1, 2, 5, 10, 20, 40]
    med = []
    for arc in arcs:
        errs = []
        for trial in range(20):
            s = make_scene(n_points=200, noise_px=0.5, arc_deg=arc, seed=trial, jitter=0.0)
            P1 = projection_matrix(s.K, s.Rs[0], s.ts[0])
            P2 = projection_matrix(s.K, s.Rs[1], s.ts[1])
            X = triangulate_dlt(P1, P2, s.obs[0], s.obs[1])
            errs.append(np.median(np.linalg.norm(X - s.X, axis=1)))
        med.append(errs)
        print(f"arc {arc} deg: median 3D error {np.median(errs):.4f}")
    a = np.array(med)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    line(ax, np.array(arcs), np.median(a, 1), 0, "DLT triangulation (σ = 0.5 px)",
         (np.percentile(a, 25, 1), np.percentile(a, 75, 1)))
    ax.set(title="3D error vs. baseline (cameras 6 units from a 3-unit object)",
           xlabel="angle between the two cameras (degrees)", ylabel="median 3D error (scene units)",
           xscale="log", yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "baseline.png"))
    plt.close(fig)


if __name__ == "__main__":
    setup()
    os.makedirs(FIG, exist_ok=True)
    normalization_study()
    degeneracy_study()
    baseline_study()
