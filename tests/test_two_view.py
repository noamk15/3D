import numpy as np
import pytest

from sfm.geometry import (to_h, skew, rotation_error_deg, direction_error_deg,
                          projection_matrix, camera_center, so3_exp, so3_log,
                          random_rotation)
from sfm.synthetic import make_scene, relative_pose
from sfm.two_view import (eight_point, sampson_error, essential_from_fundamental,
                          essential_from_pose, decompose_essential,
                          choose_pose_cheirality, epipoles, hartley_normalization,
                          eight_point_essential)
from sfm.triangulation import triangulate_dlt, refine_points, reprojection_errors


def gt_F(scene):
    R, t = relative_pose(scene.Rs[0], scene.ts[0], scene.Rs[1], scene.ts[1])
    Kinv = np.linalg.inv(scene.K)
    F = Kinv.T @ skew(t) @ R @ Kinv
    return F / np.linalg.norm(F), R, t


def same_up_to_sign(A, B):
    return min(np.linalg.norm(A - B), np.linalg.norm(A + B))


# ---------------- geometry ----------------
def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = random_rotation(rng)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


# ---------------- block 1: F ----------------
def test_hartley_normalization_properties():
    s = make_scene(seed=1)
    T, xn = hartley_normalization(s.obs[0])
    assert np.allclose(xn.mean(0), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(xn, axis=1).mean(), np.sqrt(2))


def test_eight_point_noiseless_exact():
    s = make_scene(n_points=100, seed=2)
    F = eight_point(s.obs[0], s.obs[1])
    Fgt, _, _ = gt_F(s)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert np.abs(np.einsum("ij,jk,ik->i", to_h(s.obs[1]), F, to_h(s.obs[0]))).max() < 1e-8
    assert same_up_to_sign(F, Fgt) < 1e-6


def test_eight_point_minimal_8_points():
    s = make_scene(n_points=8, seed=3)
    F = eight_point(s.obs[0], s.obs[1])
    assert same_up_to_sign(F, gt_F(s)[0]) < 1e-5


def test_rank2_and_epipoles():
    s = make_scene(noise_px=1.0, seed=4)
    F = eight_point(s.obs[0], s.obs[1])
    sv = np.linalg.svd(F, compute_uv=False)
    assert sv[2] < 1e-12 * sv[0]
    e1, e2 = epipoles(F)
    # e2 must be the image of camera centre 1 in camera 2
    C1 = camera_center(s.Rs[0], s.ts[0])
    P2 = projection_matrix(s.K, s.Rs[1], s.ts[1])
    e2_true = P2 @ np.append(C1, 1)
    assert np.degrees(np.arccos(abs(e2 @ e2_true) / np.linalg.norm(e2_true))) < 1.0


def test_normalization_helps_with_noise():
    errs_n, errs_u = [], []
    for seed in range(20):
        s = make_scene(noise_px=1.0, seed=seed)
        for normalize, errs in ((True, errs_n), (False, errs_u)):
            F = eight_point(s.obs[0], s.obs[1], normalize=normalize)
            errs.append(np.sqrt(sampson_error(F, s.obs[0], s.obs[1])).mean())
    assert np.mean(errs_n) < np.mean(errs_u)


# ---------------- block 2: E and pose ----------------
def test_essential_singular_values():
    s = make_scene(noise_px=0.5, seed=5)
    E = essential_from_fundamental(eight_point(s.obs[0], s.obs[1]), s.K)
    sv = np.linalg.svd(E, compute_uv=False)
    assert np.allclose(sv, [1, 1, 0], atol=1e-12)


def test_decomposition_four_candidates_and_cheirality():
    s = make_scene(n_points=100, seed=6)
    _, Rgt, tgt = gt_F(s)
    E = essential_from_pose(Rgt, tgt)
    cands = decompose_essential(E)
    assert len(cands) == 4
    for R, t in cands:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1)
        assert np.allclose(skew(t) @ R / np.linalg.norm(skew(t) @ R),
                           E / np.linalg.norm(E), atol=1e-8) or \
            np.allclose(skew(t) @ R / np.linalg.norm(skew(t) @ R),
                        -E / np.linalg.norm(E), atol=1e-8)
    R, t, X, counts = choose_pose_cheirality(E, s.obs[0], s.obs[1], s.K)
    assert sorted(counts) == [0, 0, 0, 100] or max(counts) == 100 and sorted(counts)[-2] < 100
    assert rotation_error_deg(R, Rgt) < 1e-6
    assert direction_error_deg(t, tgt) < 1e-6


def test_full_two_view_pose_from_pixels():
    s = make_scene(noise_px=0.5, seed=7)
    _, Rgt, tgt = gt_F(s)
    E = eight_point_essential(s.obs[0], s.obs[1], s.K)
    R, t, _, _ = choose_pose_cheirality(E, s.obs[0], s.obs[1], s.K)
    assert rotation_error_deg(R, Rgt) < 0.5
    assert direction_error_deg(t, tgt) < 2.0


# ---------------- block 3: triangulation ----------------
def test_triangulation_noiseless_exact():
    s = make_scene(seed=8)
    P1 = projection_matrix(s.K, s.Rs[0], s.ts[0])
    P2 = projection_matrix(s.K, s.Rs[1], s.ts[1])
    X = triangulate_dlt(P1, P2, s.obs[0], s.obs[1])
    assert np.abs(X - s.X).max() < 1e-8
    assert reprojection_errors(P1, X, s.obs[0]).max() < 1e-8


def test_refinement_reduces_reprojection_error():
    s = make_scene(noise_px=2.0, seed=9)
    P1 = projection_matrix(s.K, s.Rs[0], s.ts[0])
    P2 = projection_matrix(s.K, s.Rs[1], s.ts[1])
    X0 = triangulate_dlt(P1, P2, s.obs[0], s.obs[1])
    X1 = refine_points([P1, P2], s.obs, X0)
    e0 = sum((reprojection_errors(P, X0, x) ** 2).sum() for P, x in zip([P1, P2], s.obs))
    e1 = sum((reprojection_errors(P, X1, x) ** 2).sum() for P, x in zip([P1, P2], s.obs))
    assert e1 <= e0 + 1e-9
