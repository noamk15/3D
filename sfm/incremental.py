"""Incremental Structure-from-Motion.

1. Tracks: union-find over verified pairwise matches.
2. Initial pair: many inliers AND a wide enough baseline (median
   triangulation angle), to avoid the degenerate small-baseline case.
3. Two-view init: F (RANSAC) -> E -> (R,t) by cheirality -> triangulation.
4. Repeat: register the view seeing the most reconstructed tracks (PnP +
   RANSAC), triangulate new tracks, bundle adjustment, outlier filtering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import projection_matrix, camera_center, project
from .two_view import essential_from_fundamental, choose_pose_cheirality
from .robust import ransac_fundamental
from .triangulation import triangulate_nview_dlt, refine_points, triangulation_angle_deg
from .pnp import pnp_ransac
from .ba import BAProblem, ba_lm_schur, rms_reprojection


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_tracks(n_views, n_kps, matches):
    """matches[(i,j)] = (idx_i, idx_j). Returns list of tracks
    (dict view -> kp index) and kp_to_track[v] arrays."""
    uf = _UF()
    for (i, j), (a, b) in matches.items():
        for u, v in zip(a, b):
            uf.union((i, int(u)), (j, int(v)))
    groups = {}
    for node in list(uf.p.keys()):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        views = [v for v, _ in nodes]
        if len(set(views)) != len(views):        # inconsistent track: drop
            continue
        tracks.append({v: k for v, k in nodes})
    kp_to_track = [-np.ones(n, int) for n in n_kps]
    for tid, tr in enumerate(tracks):
        for v, k in tr.items():
            kp_to_track[v][k] = tid
    return tracks, kp_to_track


# ---------------------------------------------------------------------------
@dataclass
class Reconstruction:
    K: np.ndarray
    poses: dict = field(default_factory=dict)          # view -> (R, t)
    X: np.ndarray = None                               # (n_tracks, 3)
    has_X: np.ndarray = None                           # (n_tracks,) bool
    log: list = field(default_factory=list)

    def ba_problem(self, tracks, kps, max_err=None):
        views = sorted(self.poses)
        vpos = {v: k for k, v in enumerate(views)}
        tids = np.flatnonzero(self.has_X)
        tpos = {t: k for k, t in enumerate(tids)}
        ci, pi_, uv = [], [], []
        for t in tids:
            for v, k in tracks[t].items():
                if v in vpos:
                    ci.append(vpos[v]); pi_.append(tpos[t]); uv.append(kps[v][k])
        prob = BAProblem(self.K, [self.poses[v][0] for v in views],
                         [self.poses[v][1] for v in views], self.X[tids].copy(),
                         np.array(ci), np.array(pi_), np.array(uv, float))
        return prob, views, tids

    def apply_ba(self, prob, views, tids):
        for k, v in enumerate(views):
            self.poses[v] = (prob.Rs[k], prob.ts[k])
        self.X[tids] = prob.X


def _triangulate_tracks(rec, tracks, kps, cand_tids, min_angle=1.5, max_err=4.0):
    """Triangulate tracks seen by >= 2 registered views (n-view DLT + GN)."""
    added = 0
    Ps = {v: projection_matrix(rec.K, *rec.poses[v]) for v in rec.poses}
    for t in cand_tids:
        obs = [(v, k) for v, k in tracks[t].items() if v in rec.poses]
        if len(obs) < 2:
            continue
        Pl = [Ps[v] for v, _ in obs]
        xl = np.array([kps[v][k] for v, k in obs])
        X = triangulate_nview_dlt(Pl, xl)
        X = refine_points(Pl, [x[None] for x in xl], X[None], iters=5)[0]
        ok = True
        Cs = [camera_center(*rec.poses[v]) for v, _ in obs]
        ang = max(triangulation_angle_deg(Cs[a], Cs[b], X[None])[0]
                  for a in range(len(Cs)) for b in range(a + 1, len(Cs)))
        if ang < min_angle:
            ok = False
        for (v, k) in obs:
            R, tt = rec.poses[v]
            if X @ R[2] + tt[2] <= 0 or np.linalg.norm(project(rec.K, R, tt, X[None])[0] - kps[v][k]) > max_err:
                ok = False
                break
        if ok:
            rec.X[t] = X
            rec.has_X[t] = True
            added += 1
    return added


def _filter_outlier_obs(rec, tracks, kps, max_err=4.0):
    """Remove observations with large reprojection error; drop tracks left
    with < 2 observations."""
    removed = 0
    for t in np.flatnonzero(rec.has_X):
        for v in list(tracks[t].keys()):
            if v not in rec.poses:
                continue
            R, tt = rec.poses[v]
            e = np.linalg.norm(project(rec.K, R, tt, rec.X[t][None])[0] - kps[v][tracks[t][v]])
            if e > max_err or rec.X[t] @ R[2] + tt[2] <= 0:
                del tracks[t][v]
                removed += 1
        if sum(v in rec.poses for v in tracks[t]) < 2:
            rec.has_X[t] = False
    return removed


def choose_initial_pair(K, kps, matches, min_inliers=50, min_median_angle=3.0, seed=0):
    best, best_score = None, -1
    for (i, j), (a, b) in sorted(matches.items(), key=lambda kv: -len(kv[1][0])):
        if len(a) < min_inliers:
            continue
        x1, x2 = kps[i][a], kps[j][b]
        r = ransac_fundamental(x1, x2, threshold=1.5, method="msac", seed=seed)
        inl = r.inliers
        E = essential_from_fundamental(r.model, K)
        R, t, X, _ = choose_pose_cheirality(E, x1[inl], x2[inl], K)
        front = (X[:, 2] > 0) & (X @ R[2] + t[2] > 0)
        ang = triangulation_angle_deg(np.zeros(3), -R.T @ t, X[front])
        med = float(np.median(ang)) if len(ang) else 0.0
        if med < min_median_angle:
            continue
        score = int(front.sum())
        if score > best_score:
            best, best_score = (i, j, R, t, inl), score
    return best


def incremental_sfm(K, kps, matches, ba_loss="huber", ba_iters=20, min_pnp_inliers=20,
                    pnp_threshold=3.0, init_pair=None, verbose=True, seed=0):
    """Full incremental pipeline. kps[v]: (n_v,2) pixels.
    matches[(i,j)]: (idx_i, idx_j) geometrically verified matches."""
    n_views = len(kps)
    tracks, kp_to_track = build_tracks(n_views, [len(k) for k in kps], matches)
    rec = Reconstruction(K=K, X=np.zeros((len(tracks), 3)), has_X=np.zeros(len(tracks), bool))

    # ---- initial pair ------------------------------------------------------
    if init_pair is None:
        init = choose_initial_pair(K, kps, matches, seed=seed)
        if init is None:
            raise RuntimeError("no valid initial pair")
        i, j, R, t, _ = init
    else:
        i, j = init_pair
        a, b = matches[(i, j)]
        r = ransac_fundamental(kps[i][a], kps[j][b], threshold=1.5, method="msac", seed=seed)
        E = essential_from_fundamental(r.model, K)
        R, t, _, _ = choose_pose_cheirality(E, kps[i][a][r.inliers], kps[j][b][r.inliers], K)
    rec.poses[i] = (np.eye(3), np.zeros(3))
    rec.poses[j] = (R, t)
    n0 = _triangulate_tracks(rec, tracks, kps, range(len(tracks)))
    rec.log.append(f"init pair ({i},{j}): {n0} points")
    prob, views, tids = rec.ba_problem(tracks, kps)
    resba = ba_lm_schur(prob, iters=ba_iters, loss=ba_loss, fixed_cams=(views.index(i),))
    rec.apply_ba(resba.problem, views, tids)
    _filter_outlier_obs(rec, tracks, kps)
    if verbose:
        print(rec.log[-1], f"rms {resba.rms_history[-1]:.3f}px")

    # ---- add views -----------------------------------------------------------
    failed = set()
    while True:
        cands = []
        for v in range(n_views):
            if v in rec.poses or v in failed:
                continue
            tids_v = kp_to_track[v]
            ok = (tids_v >= 0)
            ok[ok] = rec.has_X[tids_v[ok]]
            cands.append((int(ok.sum()), v))
        if not cands:
            break
        cands.sort(reverse=True)
        n2d3d, v = cands[0]
        if n2d3d < min_pnp_inliers:
            break
        kidx = np.flatnonzero((kp_to_track[v] >= 0))
        kidx = kidx[rec.has_X[kp_to_track[v][kidx]]]
        X3 = rec.X[kp_to_track[v][kidx]]
        x2 = kps[v][kidx]
        R, t, inl = pnp_ransac(X3, x2, K, threshold=pnp_threshold, seed=seed)
        if R is None or inl.sum() < min_pnp_inliers:
            failed.add(v)
            rec.log.append(f"view {v}: PnP failed ({0 if R is None else inl.sum()} inliers)")
            continue
        rec.poses[v] = (R, t)
        # observations rejected by PnP are removed from the track
        for k in kidx[~inl]:
            del tracks[kp_to_track[v][k]][v]
        new_t = [tt for tt in set(kp_to_track[v][kp_to_track[v] >= 0]) if not rec.has_X[tt]]
        n_new = _triangulate_tracks(rec, tracks, kps, new_t)
        prob, views, tids = rec.ba_problem(tracks, kps)
        resba = ba_lm_schur(prob, iters=ba_iters, loss=ba_loss, fixed_cams=(views.index(i),))
        rec.apply_ba(resba.problem, views, tids)
        nrem = _filter_outlier_obs(rec, tracks, kps)
        rec.log.append(f"view {v}: PnP {inl.sum()}/{len(inl)} inliers, +{n_new} points, "
                       f"BA rms {resba.rms_history[-1]:.3f}px, removed {nrem} obs")
        if verbose:
            import resource
            print(rec.log[-1], f"[maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB]", flush=True)

    # ---- final BA ------------------------------------------------------------
    prob, views, tids = rec.ba_problem(tracks, kps)
    resba = ba_lm_schur(prob, iters=100, loss=ba_loss, fixed_cams=(views.index(i),))
    rec.apply_ba(resba.problem, views, tids)
    _filter_outlier_obs(rec, tracks, kps)
    prob, views, tids = rec.ba_problem(tracks, kps)
    rec.log.append(f"final: {len(rec.poses)} views, {rec.has_X.sum()} points, "
                   f"rms {rms_reprojection(prob):.3f}px")
    if verbose:
        print(rec.log[-1])
    return rec, tracks, kp_to_track
