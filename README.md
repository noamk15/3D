# Mini-SfM from scratch

A small Structure-from-Motion pipeline written from scratch in NumPy/SciPy
(HUJI 3D computer vision project). OpenCV is used only for image I/O, SIFT
features, and as a baseline for comparison — never for the core geometry.

**Input:** a set of images of a scene with known intrinsics K.
**Output:** a 3D point cloud, camera poses (R_i, t_i), a Three.js web viewer,
and an experimental study of robustness to outliers.

## Pipeline (built one block at a time)

| # | Block | Module | Status |
|---|---|---|---|
| 1 | Synthetic data + normalized 8-point (F) | `synthetic.py`, `two_view.py` | ⏳ |
| 2 | Essential matrix, (R,t) decomposition, cheirality | `two_view.py` | ⏳ |
| 3 | Triangulation (DLT + non-linear refinement) | `triangulation.py` | ⏳ |
| 4 | RANSAC / MSAC / LO-RANSAC, Sampson error | `robust.py` | ⏳ |
| 5 | PnP + incremental view addition | `pnp.py`, `incremental.py` | ⏳ |
| 6 | Real data (SIFT, fountain-P11) + OpenCV baseline | `features.py` | ⏳ |
| 7 | Bundle adjustment (so(3), robust losses, Schur) | `ba.py` | ⏳ |
| 8 | Three.js viewer | `viewer/` | ⏳ |
| 9 | Robustness study (outlier rate vs. error) | `experiments/` | ⏳ |
| 10 | Oral preparation | `docs/oral_prep.md` | ⏳ |

## Layout

```
sfm/          core library
tests/        unit tests on synthetic ground truth (pytest)
experiments/  reproducible scripts that generate every figure
viewer/       Three.js point cloud + camera frustums (GitHub Pages)
docs/         oral preparation (intuition FR / formal EN)
data/         datasets (not committed)
```

## Install & test

```bash
pip install -r requirements.txt
pytest -q
```
