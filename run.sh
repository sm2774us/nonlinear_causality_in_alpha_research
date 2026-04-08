#!/usr/bin/env bash
# run.sh — AlphaPod v2 one-shot build + run (Ubuntu 24.04)
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
# Requires: clang-19, python3.13, bazel-7, virtualenv

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> [1/5] Building C++26 SIMD engine (v2 regime-aware)..."
bazel build //src/alpha_engine:alpha_engine_cpp.so --config=linux

echo "==> [2/5] Deploying shared library..."
cp -f bazel-bin/src/alpha_engine/alpha_engine_cpp.so src/research/

echo "==> [3/5] Setting up Python 3.13 virtualenv..."
python3.13 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet polars numpy scipy pytest pytest-cov

echo "==> [4/5] Running tests..."
cd src/research
pytest test_alpha_pipeline.py test_nonlinear_pipeline.py -v --tb=short
cd "$ROOT"

echo "==> [5/5] Running v2 nonlinear pipeline..."
cd src/research
python3.13 alpha_pipeline.py
