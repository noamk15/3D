"""Features (OpenCV SIFT allowed) + Lowe ratio test + geometric verification
with OUR RANSAC. Also: dataset loader for the Strecha benchmark format.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
import cv2

from .robust import ransac_fundamental


def load_strecha(folder: str, scale: float = 0.5):
    """fountain-P11 style folder: images/*.jpg and gt_dense_cameras/*.camera.

    .camera file: K (3 lines), distortion (1 line), R (3 lines), C (1 line),
    image size. In this format R maps CAMERA -> WORLD and C is the centre, so in
    our convention (X_c = R' X + t'):  R' = R^T,  t' = -R^T C.
    Returns images (BGR, rescaled), K (rescaled), gt list of (R', t').
    """
    paths = sorted(glob(os.path.join(folder, "images", "*.jpg")))
    imgs, gts, K = [], [], None
    for p in paths:
        im = cv2.imread(p)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        imgs.append(im)
        cam = os.path.join(folder, "gt_dense_cameras", os.path.basename(p) + ".camera")
        vals = np.loadtxt(cam, max_rows=8)
        Kf = vals[0:3]
        R = vals[4:7]
        C = vals[7]
        Rc = R.T
        gts.append((Rc, -Rc @ C))
        if K is None:
            K = Kf.copy()
            K[:2] *= scale
    return imgs, K, gts, paths


def sift_features(imgs, n_features: int = 0, contrast_threshold: float = 0.04):
    """SIFT keypoints (pixels), descriptors and RGB colour at each keypoint.
    n_features=0 keeps all; a lower contrast threshold gives denser clouds."""
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold)
    kps, descs, colors = [], [], []
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, d = sift.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], dtype=float)
        kps.append(pts)
        descs.append(d)
        ij = np.round(pts).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, im.shape[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, im.shape[0] - 1)
        colors.append(im[ij[:, 1], ij[:, 0], ::-1].astype(np.uint8))   # RGB
    return kps, descs, colors


def ratio_test_matches(d1, d2, ratio: float = 0.8, mutual: bool = True):
    """Lowe ratio test: keep a match only if the best neighbour is clearly
    better than the second best (d1/d2 < ratio) -> rejects ambiguous matches
    (repeated textures). Optional mutual (cross-check) consistency."""
    # FLANN (randomized kd-trees): approximate nearest neighbours, much faster
    # than brute force for ~10^4 descriptors per image.
    bf = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    m12 = bf.knnMatch(d1, d2, k=2)
    good = {}
    for m in m12:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good[m[0].queryIdx] = m[0].trainIdx
    if mutual:
        m21 = bf.knnMatch(d2, d1, k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
        good = {a: b for a, b in good.items() if back.get(b) == a}
    a = np.array(list(good.keys()), int)
    b = np.array(list(good.values()), int)
    return a, b


def match_all_pairs(kps, descs, ratio=0.8, ransac_thresh=1.5, min_inliers=40,
                    method="msac", window=None, verbose=True):
    """All pairs (or pairs within `window` indices): ratio test then
    geometric verification with our F-RANSAC. Returns {(i,j): (a,b)} and stats."""
    n = len(kps)
    matches, stats = {}, {}
    for i in range(n):
        for j in range(i + 1, n):
            if window is not None and j - i > window:
                continue
            a, b = ratio_test_matches(descs[i], descs[j], ratio)
            if len(a) < min_inliers:
                continue
            r = ransac_fundamental(kps[i][a], kps[j][b], threshold=ransac_thresh,
                                   method=method, seed=i * n + j)
            ninl = int(r.inliers.sum())
            stats[(i, j)] = (len(a), ninl, r.iterations)
            if ninl >= min_inliers:
                matches[(i, j)] = (a[r.inliers], b[r.inliers])
            if verbose:
                print(f"pair ({i:2d},{j:2d}): {len(a):5d} ratio-test matches, "
                      f"{ninl:5d} inliers, {r.iterations} iters")
    return matches, stats
