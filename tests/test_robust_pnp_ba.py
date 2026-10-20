import numpy as np

from sfm.geometry import (rotation_error_deg, direction_error_deg, projection_matrix,
                          umeyama_similarity, camera_center, so3_exp)
from sfm.synthetic import make_scene, add_outliers, relative_pose
from sfm.two_view import eight_point, essential_from_fundamental, choose_pose_cheirality
from sfm.robust import ransac_fundamental, ransac_iterations, robust_weights
from sfm.pnp import pnp_dlt, pnp_ransac
from sfm.ba import BAProblem, ba_lm_schur, ba_scipy, rms_reprojection, solve_normal_equations_dense
from sfm.incremental import incremental_sfm


# ---------------- block 4: robust ----------------
def test_ransac_iteration_formula():
    # classic table (Hartley-Zisserman): s=8, w=0.5, p=0.99 -> 1177
    assert ransac_iterations(0.99, 0.5, 8) == 1177
    assert ransac_iterations(0.99, 1.0, 8) == 1


def test_ransac_recovers_pose_with_50pct_outliers():
    s = make_scene(n_points=300, noise_px=0.5, seed=11)
    x2, inl_true = add_outliers(s.obs[1], 0.5, seed=11)
    Rgt, tgt = relative_pose(s.Rs[0], s.ts[0], s.Rs[1], s.ts[1])
    for method in ("ransac", "msac", "lo"):
        r = ransac_fundamental(s.obs[0], x2, threshold=1.5, method=method, seed=0)
        E = essential_from_fundamental(r.model, s.K)
        R, t, _, _ = choose_pose_cheirality(E, s.obs[0][r.inliers], x2[r.inliers], s.K)
        assert rotation_error_deg(R, Rgt) < 1.0, method
        assert direction_error_deg(t, tgt) < 3.0, method
        # precision of inlier classification
        assert (r.inliers & ~inl_true).sum() <= 0.05 * r.inliers.sum()


def test_least_squares_breaks_with_outliers():
    s = make_scene(n_points=300, noise_px=0.5, seed=12)
    x2, _ = add_outliers(s.obs[1], 0.3, seed=12)
    Rgt, _ = relative_pose(s.Rs[0], s.ts[0], s.Rs[1], s.ts[1])
    E = essential_from_fundamental(eight_point(s.obs[0], x2), s.K)
    R, _, _, _ = choose_pose_cheirality(E, s.obs[0], x2, s.K)
    assert rotation_error_deg(R, Rgt) > 2.0


def test_robust_weights():
    r = np.array([0.0, 0.5, 2.0, 100.0])
    assert np.allclose(robust_weights(r, "l2"), 1)
    assert robust_weights(r, "huber")[3] == 0.01
    assert robust_weights(r, "cauchy")[3] < 1e-3


# ---------------- block 5: PnP ----------------
def test_pnp_dlt_noiseless():
    s = make_scene(n_points=50, seed=13)
    R, t = pnp_dlt(s.X, s.obs[1], s.K)
    assert rotation_error_deg(R, s.Rs[1]) < 1e-6
    assert np.linalg.norm(t - s.ts[1]) < 1e-6


def test_pnp_ransac_noisy_outliers():
    s = make_scene(n_points=200, noise_px=1.0, seed=14)
    x, _ = add_outliers(s.obs[1], 0.4, seed=14)
    R, t, inl = pnp_ransac(s.X, x, s.K, threshold=3.0)
    assert rotation_error_deg(R, s.Rs[1]) < 0.5
    assert np.linalg.norm(camera_center(R, t) - camera_center(s.Rs[1], s.ts[1])) < 0.05


# ---------------- block 7: BA ----------------
def _perturbed_problem(seed=15, n_cams=5, n_points=80, noise=0.5):
    s = make_scene(n_points=n_points, n_cams=n_cams, arc_deg=60, noise_px=noise, seed=seed)
    rng = np.random.default_rng(seed)
    Rs = [s.Rs[0]] + [so3_exp(0.02 * rng.normal(size=3)) @ R for R in s.Rs[1:]]
    ts = [s.ts[0], s.ts[1]] + [t + 0.05 * rng.normal(size=3) for t in s.ts[2:]]
    X = s.X + 0.05 * rng.normal(size=s.X.shape)
    ci = np.repeat(np.arange(n_cams), n_points)
    pi_ = np.tile(np.arange(n_points), n_cams)
    uv = np.concatenate(s.obs)
    return s, BAProblem(s.K, Rs, ts, X, ci, pi_, uv)


def test_schur_step_equals_full_step():
    _, prob = _perturbed_problem(n_cams=3, n_points=15)
    d_full, _ = solve_normal_equations_dense(prob, lam=1e-3)
    # one LM iteration of the Schur solver from the same point, no scale fix
    r = ba_lm_schur(prob, iters=1, lam0=1e-3, fix_scale=False)
    # reconstruct the applied point step
    dX = (r.problem.X - prob.X).ravel()
    nc = len(prob.Rs)
    assert np.allclose(dX, d_full[6 * nc:], atol=1e-8)


def test_ba_converges_noiseless_to_zero():
    s, prob = _perturbed_problem(noise=0.0)
    r = ba_lm_schur(prob, iters=50)
    assert r.rms_history[-1] < 1e-6
    assert r.rms_history[0] > 1.0


def test_ba_handwritten_matches_scipy():
    _, prob = _perturbed_problem(noise=0.5)
    a = ba_lm_schur(prob, iters=100)
    b = ba_scipy(prob)
    assert abs(a.rms_history[-1] - b.rms_history[-1]) < 1e-3


# ---------------- block 5: incremental ----------------
def test_incremental_sfm_synthetic():
    s = make_scene(n_points=300, n_cams=6, arc_deg=70, noise_px=0.5, seed=16)
    n = len(s.X)
    kps = s.obs
    matches = {(i, j): (np.arange(n), np.arange(n)) for i in range(6) for j in range(i + 1, 6)}
    rec, tracks, _ = incremental_sfm(s.K, kps, matches, verbose=False)
    assert len(rec.poses) == 6
    views = sorted(rec.poses)
    Cest = np.array([camera_center(*rec.poses[v]) for v in views])
    Cgt = np.array([camera_center(s.Rs[v], s.ts[v]) for v in views])
    sc, R, t = umeyama_similarity(Cest, Cgt)
    aligned = sc * Cest @ R.T + t
    assert np.abs(aligned - Cgt).max() < 0.05
    for v in views:
        Rv = rec.poses[v][0]
        # relative rotation to view 0 is gauge-invariant
        Rrel = Rv @ rec.poses[views[0]][0].T
        Rrel_gt = s.Rs[v] @ s.Rs[views[0]].T
        assert rotation_error_deg(Rrel, Rrel_gt) < 0.3


# ---------------- analytic Jacobians vs. finite differences ----------------
def test_projection_jacobians_finite_differences():
    from sfm.geometry import project_with_jacobians, project, random_rotation
    rng = np.random.default_rng(17)
    K = np.array([[800.0, 0, 500], [0, 810, 380], [0, 0, 1]])
    R = random_rotation(rng)
    t = np.array([0.1, 0.2, 6.0])
    X = rng.normal(size=(20, 3))
    uv, Jp, Jx = project_with_jacobians(K, R, t, X)
    eps = 1e-6
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        num = (project(K, so3_exp(d[:3]) @ R, t + d[3:], X)
               - project(K, so3_exp(-d[:3]) @ R, t - d[3:], X)) / (2 * eps)
        assert np.allclose(num, Jp[:, :, k], rtol=1e-5, atol=1e-4)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        num = (project(K, R, t, X + d) - project(K, R, t, X - d)) / (2 * eps)
        assert np.allclose(num, Jx[:, :, k], rtol=1e-5, atol=1e-4)
