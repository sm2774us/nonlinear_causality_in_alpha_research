// src/alpha_engine/MacroAlphaEngine.ixx
// Copyright 2026 AlphaPod Contributors. All Rights Reserved.
//
// C++26 module: MacroAlphaEngine v2 — Regime-Conditional Pipeline
//
// UPGRADE from v1: accepts a Regime enum and routes each pipeline step
// to the regime-aware RiskKernel functions.
//
// v1 pipeline (WRONG — superposition):
//   cross_sectional_zscore(alpha)            ← Pearson z, fixed clip
//   apply_vol_scaling(alpha, vol, sigma*)    ← fixed sigma*=10%
//   apply_circuit_breaker(alpha, cap=0.20)   ← fixed cap
//
// v2 pipeline (CORRECT — regime-conditional nonlinear):
//   rank_zscore_simd(alpha, clip[regime])    ← rank-based, clip varies by regime
//   apply_vol_scaling_regime(alpha, vol, R)  ← sigma* from kRegimeTable[R]
//   apply_nonlinear_interaction_cap(alpha, vol, sigma*, cap[R]) ← vol-adjusted cap
//   apply_circuit_breaker_regime(alpha, R)   ← hard floor, regime-specific
//
// Thread safety: MacroAlphaEngine v2 is still stateless after construction;
// the regime is passed per-call, not stored. Multiple Python threads may call
// run_pipeline() concurrently on different (data, regime) pairs.
//
// Example (C++):
//   alpha_pod::MacroAlphaEngine engine{};
//   engine.run_pipeline(alpha, vol, alpha_pod::Regime::STRESS);
module; // START OF GLOBAL MODULE FRAGMENT
#include "MacroAlphaEngine.h" // This includes RiskKernel.h via preprocessor
#include "RiskKernel.h"

export module AlphaPod.MacroAlphaEngine;

import std;
export import AlphaPod.RiskKernel; // This is the Module-level instruction

namespace alpha_pod {

// ── Constructor ──────────────────────────────────────────────────────────────
MacroAlphaEngine::MacroAlphaEngine(float vol_target, float weight_cap)
    : vol_target_{vol_target}, weight_cap_{weight_cap} {}

// ── v2 Pipeline (regime-conditional) ─────────────────────────────────────
std::expected<void, std::string>
MacroAlphaEngine::run_pipeline(
    std::span<float>       alpha,
    std::span<const float> f_vol,
    Regime                 regime
) const noexcept {
    // Here we use the table from the imported RiskKernel module
    const auto& params = kRegimeTable[static_cast<int>(regime)];

    // const float clip = kRegimeTable[static_cast<int>(regime)].zscore_clip;
    // const float vt   = kRegimeTable[static_cast<int>(regime)].vol_target;
    // const float cap  = kRegimeTable[static_cast<int>(regime)].weight_cap;
    const float clip = params.zscore_clip;
    const float vt   = params.vol_target;
    const float cap  = params.weight_cap;

    // Step 1: Rank-based z-score (regime-specific clip)
    if (auto r = RiskKernel::rank_zscore_simd(alpha, clip); !r) return r;

    // Step 2: Regime-conditional vol scaling
    if (auto r = RiskKernel::apply_vol_scaling_regime(alpha, f_vol, regime); !r) return r;

    // Step 3: Nonlinear interaction cap (alpha × vol → tighter cap for high-vol assets)
    if (auto r = RiskKernel::apply_nonlinear_interaction_cap(alpha, f_vol, vt, cap); !r) return r;

    // Step 4: Hard regime-specific circuit breaker
    if (auto r = RiskKernel::apply_circuit_breaker_regime(alpha, regime); !r) return r;

    return {};
}

// ── v1 legacy API (backward compatible) ───────────────────────────────────
// Implementation of the 2-arg version (delegates to the 3-arg version)
std::expected<void, std::string>
MacroAlphaEngine::run_pipeline(
    std::span<float>       alpha,
    std::span<const float> f_vol
) const noexcept {
    return run_pipeline(alpha, f_vol, Regime::TRANSITION);
}

// ── Getters for nanobind ───────────────────────────────────────────────────
float MacroAlphaEngine::vol_target() const noexcept { return vol_target_; }
float MacroAlphaEngine::weight_cap() const noexcept { return weight_cap_; }

} // namespace alpha_pod
