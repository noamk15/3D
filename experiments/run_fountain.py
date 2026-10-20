"""Full pipeline on fountain-P11 + evaluation against ground truth + OpenCV
baseline. Writes experiments/out/fountain_*.{json,ply,txt} and the viewer data.

    python -m experiments.run_fountain [--scale 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import cv2

from sfm.features import load_strecha, sift_features, match_all_pairs
from sfm.incremental import incremental_sfm
from sfm.geometry import (camera_center, rotation_error_deg, direction_error_deg,
                          umeyama_similarity)
from sfm.robust import ransac_fundamental
from sfm.two_view import essential_from_fundamental, choose_pose_cheirality
from sfm.synthetic import relative_pose
from sfm.ba import rms_reprojection
from sfm.io import write_ply, write_viewer_json

OUT = os.path.join(os.path.dirname(__file__), "out")


def align_and_evaluate(rec, gts):
    views = sorted(rec.poses)
    Ce = np.array([camera_center(*rec.poses[v]) for v in views])
    Cg = np.array([camera_center(*gts[v]) for v in views])
    s, Ra, ta = umeyama_similarity(Ce, Cg)
    Ce_al = s * Ce @ Ra.T + ta
    pos_err = np.linalg.norm(Ce_al - Cg, axis=1)
    # rotation error after alignment: R_est_aligned = R_est Ra^T
    rot_err = [rotation_error_deg(rec.poses[v][0] @ Ra.T, gts[v][0]) for v in views]
    scene_size = np.linalg.norm(Cg - Cg.mean(0), axis=1).max()
    return dict(views=views, rot_err_deg=rot_err, pos_err=pos_err.tolist(),
                pos_err_rel=(pos_err / scene_size).tolist(), sim=(s, Ra, ta))


def two_view_comparison(kps, matches, K, gts):
    """Relative pose of each verified pair: ours vs OpenCV (5-point + RANSAC)."""
    rows = []
    for (i, j), (a, b) in sorted(matches.items()):
        x1, x2 = kps[i][a], kps[j][b]
        Rgt, tgt = relative_pose(*gts[i], *gts[j])
        t0 = time.perf_counter()
        r = ransac_fundamental(x1, x2, threshold=1.0, method="msac", seed=0)
        E = essential_from_fundamental(r.model, K)
        R, t, _, _ = choose_pose_cheirality(E, x1[r.inliers], x2[r.inliers], K)
        t_ours = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ecv, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, Rcv, tcv, _ = cv2.recoverPose(Ecv[:3], x1, x2, K, mask=mask)
        t_cv = time.perf_counter() - t0
        rows.append(dict(pair=[i, j], n=len(a),
                         ours_rot=rotation_error_deg(R, Rgt), ours_t=direction_error_deg(t, tgt),
                         cv_rot=rotation_error_deg(Rcv, Rgt), cv_t=direction_error_deg(tcv.ravel(), tgt),
                         ours_time=t_ours, cv_time=t_cv))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fountain-P11")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--contrast", type=float, default=0.01, help="SIFT contrast threshold")
    ap.add_argument("--prefix", default="fountain")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    imgs, K, gts, _ = load_strecha(args.data, args.scale)
    h, w = imgs[0].shape[:2]
    t0 = time.perf_counter()
    kps, descs, cols = sift_features(imgs, contrast_threshold=args.contrast)
    print('keypoints per image:', [len(k) for k in kps])
    matches, stats = match_all_pairs(kps, descs, verbose=False)
    t_feat = time.perf_counter() - t0
    print(f"features+matching: {t_feat:.1f}s, {len(matches)} verified pairs")

    t0 = time.perf_counter()
    rec, tracks, _ = incremental_sfm(K, kps, matches, ba_loss="huber")
    t_sfm = time.perf_counter() - t0
    ev = align_and_evaluate(rec, gts)
    prob, views, tids = rec.ba_problem(tracks, kps)
    rms = rms_reprojection(prob)
    n_obs = len(prob.uv)
    print(f"SfM: {t_sfm:.1f}s, rms {rms:.3f}px, {len(tids)} points, {n_obs} observations")
    print("rotation error (deg) per view:", np.round(ev["rot_err_deg"], 3))
    print("centre error (% of scene size):", np.round(100 * np.array(ev["pos_err_rel"]), 3))

    tv = two_view_comparison(kps, matches, K, gts)
    ours = np.array([[r["ours_rot"], r["ours_t"]] for r in tv])
    cvv = np.array([[r["cv_rot"], r["cv_t"]] for r in tv])
    print(f"two-view, median over {len(tv)} pairs: ours rot {np.median(ours[:,0]):.3f} deg, "
          f"t {np.median(ours[:,1]):.3f} deg | OpenCV rot {np.median(cvv[:,0]):.3f}, t {np.median(cvv[:,1]):.3f}")

    # colours: from the first observation of each track
    colors = []
    for t in tids:
        v, k = next(iter(tracks[t].items()))
        colors.append(cols[v][k])
    colors = np.array(colors)
    # express everything in the GT frame (metric, oriented) for the viewer
    s, Ra, ta = ev["sim"]
    Xw = s * rec.X[tids] @ Ra.T + ta
    poses_w = {}
    for v in views:
        R, t = rec.poses[v]
        Rw = R @ Ra.T
        Cw = s * camera_center(R, t) @ Ra.T + ta
        poses_w[v] = (Rw, -Rw @ Cw)
    write_ply(os.path.join(OUT, f"{args.prefix}.ply"), Xw, colors)
    meta = dict(dataset="fountain-P11", n_views=len(views), n_points=int(len(tids)),
                rms_px=round(rms, 3), median_rot_err_deg=round(float(np.median(ev["rot_err_deg"])), 4))
    write_viewer_json(os.path.join(OUT, f"{args.prefix}_scene.json"), K, poses_w, Xw, colors, w, h,
                      gt_poses={v: gts[v] for v in views}, meta=meta)
    summary = dict(scale=args.scale, K=K.tolist(), time_features_s=t_feat, time_sfm_s=t_sfm,
                   n_verified_pairs=len(matches), rms_px=rms, n_points=int(len(tids)),
                   n_observations=int(n_obs), log=rec.log,
                   rot_err_deg=ev["rot_err_deg"], pos_err_rel=ev["pos_err_rel"],
                   two_view=tv)
    with open(os.path.join(OUT, f"{args.prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    # cache for the other experiments (BA convergence, real-data outliers)
    import pickle
    with open(os.path.join(OUT, f"{args.prefix}_cache.pkl"), "wb") as f:
        pickle.dump(dict(K=K, kps=kps, matches=matches, gts=gts, poses=rec.poses,
                         X=rec.X, has_X=rec.has_X, tracks=tracks), f)


if __name__ == "__main__":
    main()
