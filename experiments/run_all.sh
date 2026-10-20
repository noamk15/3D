#!/usr/bin/env bash
# Reproduce every number and figure of the project (fountain-P11 must be in data/).
set -e
cd "$(dirname "$0")/.."
python3 -u -m experiments.run_fountain "$@"        # add --reuse-matches to skip SIFT/matching
python3 viewer/build.py experiments/out/fountain_scene.json
python3 -u -m experiments.conditioning
python3 -u -m experiments.ba_convergence
python3 -u -m experiments.robustness
