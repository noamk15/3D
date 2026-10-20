"""Block 9 — robustness study.

A. Two-view relative pose vs. outlier ratio:
   plain least squares (8-point on everything) / RANSAC / MSAC / LO-RANSAC.
   Synthetic (ground truth) and real (fountain pair with injected outliers).
B. Bundle adjustment with corrupted observations: L2 vs Huber vs Cauchy.

    python -m experiments.robustness
Figures -> docs/figures/, numbers -> experiments/out/robustness.json
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from sfm.geometry import (rotation_error_deg, direction_error_deg, so3_exp, camera_center,
                          umeyama_similarity)
from sfm.synthetic import make_scene, add_outliers, relative_pose
from sfm.two_view import eight_point, essential_from_fundamental, choose_pose_cheirality
from sfm.robust import ransac_fundamental
from sfm.ba import BAProblem, ba_lm_schur, residuals
from experiments.plotstyle import setup, line, plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "docs", "figures")
OUT = os.path.join(HERE, "out")
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
METHODS = ["lstsq", "ransac", "msac", "lo"]
LABELS = {"lstsq": "Least squares (no RANSAC)", "ransac": "RANSAC", "msac": "MSAC",
          "lo": "LO-RANSAC"}


def estimate(method, x1, x2, K, seed):
    t0 = time.perf_counter()
    if method == "lstsq":
        F, inl, iters = eight_point(x1, x2), np.ones(len(x1), bool), 1
    else:
        r = ransac_fundamental(x1, x2, threshold=1.5, method=method, seed=seed,
                               max_iters=20_000)
        F, inl, iters = r.model, r.inliers, r.iterations
    E = essential_from_fundamental(F, K)
    R, t, _, _ = choose_pose_cheirality(E, x1[inl], x2[inl], K)
    return R, t, iters, time.perf_counter() - t0


def two_view_study(get_data, n_trials):
    res = {m: {k: [] for k in ("rot", "t", "time", "iters")} for m in METHODS}
    for ratio in RATIOS:
        acc = {m: {k: [] for k in ("rot", "t", "time", "iters")} for m in METHODS}
        for trial in range(n_trials):
            x1, x2, K, Rgt, tgt = get_data(ratio, trial)
            for m in METHODS:
                R, t, it, dt = estimate(m, x1, x2, K, seed=trial)
                acc[m]["rot"].append(rotation_error_deg(R, Rgt))
                acc[m]["t"].append(direction_error_deg(t, tgt))
                acc[m]["time"].append(dt)
                acc[m]["iters"].append(it)
        for m in METHODS:
            for k in acc[m]:
                res[m][k].append(acc[m][k])
        print(f"ratio {ratio:.1f}: " + ", ".join(
            f"{m} rot {np.median(acc[m]['rot']):.3f}" for m in METHODS))
    return res


def plot_two_view(res, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    pct = 100 * np.array(RATIOS)
    for k, m in enumerate(METHODS):
        for ax, key in zip(axes, ("rot", "t", "time")):
            arr = np.array(res[m][key])
            if key == "time":
                arr = arr * 1000
            med = np.median(arr, axis=1)
            q1, q3 = np.percentile(arr, 25, axis=1), np.percentile(arr, 75, axis=1)
            line(ax, pct, med, k, LABELS[m], (q1, q3))
    axes[0].set(title="Rotation error", xlabel="Outlier ratio (%)", ylabel="degrees (median, IQR)",
                yscale="log")
    axes[1].set(title="Translation direction error", xlabel="Outlier ratio (%)",
                ylabel="degrees (median, IQR)", yscale="log")
    axes[2].set(title="Run time", xlabel="Outlier ratio (%)", ylabel="ms (median)", yscale="log")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. BA with outlier observations
# ---------------------------------------------------------------------------
def ba_study(n_trials=10, n_cams=8, n_points=250):
    losses = ["l2", "huber", "cauchy"]
    ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    out = {l: {"rot": [], "pos": [], "rms_inl": [], "time": []} for l in losses}
    for ratio in ratios:
        acc = {l: {"rot": [], "pos": [], "rms_inl": [], "time": []} for l in losses}
        for trial in range(n_trials):
            s = make_scene(n_points=n_points, n_cams=n_cams, arc_deg=80, noise_px=0.5,
                           seed=1000 + trial)
            rng = np.random.default_rng(trial)
            ci = np.repeat(np.arange(n_cams), n_points)
            pi_ = np.tile(np.arange(n_points), n_cams)
            uv = np.concatenate(s.obs).copy()
            bad = rng.random(len(uv)) < ratio
            uv[bad] += rng.normal(scale=40.0, size=(bad.sum(), 2))   # gross errors
            Rs = [s.Rs[0]] + [so3_exp(0.03 * rng.normal(size=3)) @ R for R in s.Rs[1:]]
            ts = [s.ts[0], s.ts[1]] + [t + 0.1 * rng.normal(size=3) for t in s.ts[2:]]
            X0 = s.X + 0.1 * rng.normal(size=s.X.shape)
            prob = BAProblem(s.K, Rs, ts, X0, ci, pi_, uv)
            Cg = np.array([camera_center(R, t) for R, t in zip(s.Rs, s.ts)])
            for l in losses:
                r = ba_lm_schur(prob, iters=100, loss=l, loss_scale=2.0)
                p = r.problem
                Ce = np.array([camera_center(R, t) for R, t in zip(p.Rs, p.ts)])
                sc, Ra, ta = umeyama_similarity(Ce, Cg)
                pos = np.linalg.norm(sc * Ce @ Ra.T + ta - Cg, axis=1).mean()
                rot = np.mean([rotation_error_deg(R @ Ra.T, Rg) for R, Rg in zip(p.Rs, s.Rs)])
                res_ = np.linalg.norm(residuals(p), axis=1)
                acc[l]["rot"].append(rot)
                acc[l]["pos"].append(pos)
                acc[l]["rms_inl"].append(float(np.sqrt(np.mean(res_[~bad] ** 2))))
                acc[l]["time"].append(r.time_s)
        for l in losses:
            for k in acc[l]:
                out[l][k].append(acc[l][k])
        print(f"BA ratio {ratio:.2f}: " + ", ".join(
            f"{l} rot {np.median(acc[l]['rot']):.3f} pos {np.median(acc[l]['pos']):.4f}" for l in losses))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    pct = 100 * np.array(ratios)
    names = {"l2": "Quadratic (L2)", "huber": "Huber", "cauchy": "Cauchy"}
    for k, l in enumerate(losses):
        for ax, key in zip(axes, ("rot", "pos", "rms_inl")):
            arr = np.array(out[l][key])
            line(ax, pct, np.median(arr, 1), k, names[l],
                 (np.percentile(arr, 25, 1), np.percentile(arr, 75, 1)))
    axes[0].set(title="Camera rotation error", xlabel="Corrupted observations (%)",
                ylabel="degrees (median, IQR)", yscale="log")
    axes[1].set(title="Camera centre error", xlabel="Corrupted observations (%)",
                ylabel="scene units (median, IQR)", yscale="log")
    axes[2].set(title="RMS reprojection on clean obs.", xlabel="Corrupted observations (%)",
                ylabel="pixels", yscale="log")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Bundle adjustment with gross observation errors (8 cameras, σ = 0.5 px, outliers ~ N(0, 40 px))",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ba_robust_losses.png"))
    plt.close(fig)
    return dict(ratios=ratios, results=out)


def main():
    setup()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    summary = {}

    # A1 synthetic
    def synth(ratio, trial):
        s = make_scene(n_points=300, noise_px=0.5, seed=trial, arc_deg=30)
        x2, _ = add_outliers(s.obs[1], ratio, seed=trial)
        R, t = relative_pose(s.Rs[0], s.ts[0], s.Rs[1], s.ts[1])
        return s.obs[0], x2, s.K, R, t
    print("== two-view synthetic ==")
    res = two_view_study(synth, n_trials=30)
    plot_two_view(res, "Two-view pose vs. outliers — synthetic (300 matches, σ = 0.5 px, 30 trials)",
                  "two_view_outliers_synthetic.png")
    summary["two_view_synthetic"] = res

    # A2 real: fountain pair (4,5) verified inliers + injected random matches
    cache = os.path.join(OUT, "fountain_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            C = pickle.load(f)
        K, kps, gts = C["K"], C["kps"], C["gts"]
        i, j = 4, 5
        a, b = C["matches"][(i, j)]
        Rgt, tgt = relative_pose(*gts[i], *gts[j])

        def real(ratio, trial):
            rng = np.random.default_rng(trial)
            n_in = len(a)
            n_out = int(round(ratio / (1 - ratio) * n_in)) if ratio < 1 else 0
            ra = rng.integers(0, len(kps[i]), n_out)
            rb = rng.integers(0, len(kps[j]), n_out)
            x1 = np.vstack([kps[i][a], kps[i][ra]])
            x2 = np.vstack([kps[j][b], kps[j][rb]])
            return x1, x2, K, Rgt, tgt
        print("== two-view real (fountain 4-5) ==")
        resr = two_view_study(real, n_trials=15)
        plot_two_view(resr, f"Two-view pose vs. outliers — fountain-P11 pair (4,5), {len(a)} true matches + random wrong matches",
                      "two_view_outliers_fountain.png")
        summary["two_view_fountain"] = resr

    print("== BA robust losses ==")
    summary["ba"] = ba_study()
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
