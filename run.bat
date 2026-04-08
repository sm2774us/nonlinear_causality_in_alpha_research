@echo off
REM run.bat — AlphaPod v2 one-shot build + run (Windows 11)
REM Copyright 2026 AlphaPod Contributors. All Rights Reserved.
REM Requires: Visual Studio 2022 Build Tools, Python 3.13, Bazel 7.

setlocal enabledelayedexpansion

echo =^> [1/5] Building C++26 SIMD engine (v2 regime-aware)...
bazel build //src/alpha_engine:alpha_engine_cpp.so --config=windows
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )

echo =^> [2/5] Deploying Python extension...
copy /Y bazel-bin\src\alpha_engine\alpha_engine_cpp.so src\research\alpha_engine_cpp.pyd
if errorlevel 1 ( echo COPY FAILED & exit /b 1 )

echo =^> [3/5] Installing Python dependencies...
python -m pip install --quiet polars numpy scipy pytest pytest-cov
if errorlevel 1 ( echo PIP FAILED & exit /b 1 )

echo =^> [4/5] Running tests...
pytest src\research\test_alpha_pipeline.py src\research\test_nonlinear_pipeline.py -v --tb=short
if errorlevel 1 ( echo TESTS FAILED & exit /b 1 )

echo =^> [5/5] Running v2 nonlinear pipeline...
cd src\research
python alpha_pipeline.py
if errorlevel 1 ( echo PIPELINE FAILED & exit /b 1 )

echo.
echo AlphaPod v2 run complete.
pause
