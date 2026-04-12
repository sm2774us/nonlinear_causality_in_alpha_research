# AlphaPod v2 — Nonlinear Causality Systematic Macro Pipeline

> **C++26 · Python 3.13 · Bazel 7 · MAX Engine · Polars · nanobind**  
> Addresses Ma & Prosperino (2023): nonlinear causality is systematically underestimated by linear models.

---

## Repository Layout

```
alphapod/
├── MODULE.bazel                        # Bzlmod dependency manifest
├── .bazelrc                            # C++26/SIMD/Python build flags
├── .github/workflows/ci.yml            # Ubuntu + Windows CI matrix
├── Dockerfile                          # Multi-stage reproducible container
├── run.sh / run.bat                    # One-shot build + run scripts
├── requirements_lock.txt               # Locked pip deps
├── SOLUTION_EXPLANATION.md             # Full mathematical deep-dive (MathJax + Mermaid)
├── models/                             # AlphaPod v2 Model Artifacts
├── research/research.{bib|tex|pdf}     # Academic paper (LaTeX)
├── notesbooks/research.ipynb           # Backtesting notebook (KPIs, black swan tests)
└── src/
    ├── alpha_engine/                   # C++26 SIMD risk kernel
    │   ├── BUILD.bazel
    │   ├── RiskKernel.ixx              # rank_zscore_simd, vol_scaling_regime, nonlinear_cap
    │   ├── MacroAlphaEngine.ixx        # Top-level engine: run_pipeline(alpha, vol, regime)
    │   ├── bindings.cpp                # nanobind bridge; exposes Regime enum to Python
    │   └── alpha_engine_test.cpp       # GoogleTest v2: regime ordering, NaN guards
    └── research/                       # Python 3.13 research pipeline
        ├── BUILD.bazel
        ├── config.py                   # Frozen PodConfig (v2 + causal/regime params)
        ├── causality.py                # KSG Transfer Entropy + AAFT + CCM
        ├── regime.py                   # Gaussian HMM: CALM/TRANSITION/STRESS
        ├── features.py                 # Rank z-score, interactions, regime-gating
        ├── inference.py                # MAX Engine wrapper / NonlinearFallbackModel
        ├── alpha_pipeline.py           # v2 orchestration entry-point
        ├── test_alpha_pipeline.py      # pytest baseline suite (v1+v2 compatibility)
        └── test_nonlinear_pipeline.py  # pytest v2 suite (100% coverage)
```

## Quick Start (Ubuntu 24.04)

```bash
chmod +x run.sh && ./run.sh
```

## Manual Steps

```bash
# Build C++26 SIMD engine
bazel build //src/alpha_engine:alpha_engine_cpp.so --config=linux
cp bazel-bin/src/alpha_engine/alpha_engine_cpp.so src/research/

# Python deps
python3.13 -m venv .venv && source .venv/bin/activate
pip install polars numpy scipy pytest pytest-cov

# Run tests
pytest src/research/test_alpha_pipeline.py src/research/test_nonlinear_pipeline.py -v

# Run v2 pipeline
python3.13 src/research/alpha_pipeline.py
```

## v1 → v2 Upgrade Summary

| Flaw | Fix |
|------|-----|
| Linear superposition `w=[1,1,-1]` | `NonlinearFallbackModel`: sigmoid-gated base + interactions |
| Pearson z-score | Rank z-score (probit, Spearman-consistent) |
| No causal features | KSG-TE + AAFT surrogates + CCM → `f_te_net_cause`, `f_ccm_driver` |
| Fixed single-regime risk | Gaussian HMM → 3 regimes → regime-conditional vol-target, cap |

## Reference

Ma & Prosperino (2023). *Nonlinear Causality in Financial Markets.*  
Copyright 2026 AlphaPod Contributors. All Rights Reserved.
