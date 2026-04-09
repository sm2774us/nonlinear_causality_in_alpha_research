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

export module AlphaPod.MacroAlphaEngine;

import AlphaPod.RiskKernel;
import std;

namespace alpha_pod {

/// @brief Regime-conditional risk-transformation pipeline.
export class MacroAlphaEngine {
public:
    /// Default-constructed engine uses kRegimeTable for all parameters.
    /// Explicit vol_target / weight_cap are ONLY used as legacy fallback
    /// when run_pipeline() is called without a regime argument.
    explicit MacroAlphaEngine(
        float vol_target = kRegimeTable[1].vol_target,
        float weight_cap = kRegimeTable[1].weight_cap
    ) : vol_target_{vol_target}, weight_cap_{weight_cap} {}

    // ── v2 Pipeline (regime-conditional) ─────────────────────────────────────

    /// @brief Runs the NONLINEAR regime-conditional risk pipeline in-place.
    ///
    /// Steps:
    ///   1. rank_zscore_simd      — rank-based z-score; clip tightened in STRESS
    ///   2. apply_vol_scaling_regime  — vol-target from kRegimeTable[regime]
    ///   3. apply_nonlinear_interaction_cap — per-asset effective cap ∝ vol
    ///   4. apply_circuit_breaker_regime    — hard floor from kRegimeTable[regime]
    ///
    /// @param alpha   [in/out] Raw model scores → final weights.
    /// @param f_vol   [in]     Realised annualised vol per asset.
    /// @param regime  Current market regime (Python RegimeEngine output).
    [[nodiscard]] std::expected<void, std::string>
    run_pipeline(
        std::span<float>       alpha,
        std::span<const float> f_vol,
        Regime                 regime = Regime::TRANSITION
    ) const noexcept {
        const float clip = kRegimeTable[static_cast<int>(regime)].zscore_clip;
        const float vt   = kRegimeTable[static_cast<int>(regime)].vol_target;
        const float cap  = kRegimeTable[static_cast<int>(regime)].weight_cap;

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

    /// @brief v1 pipeline — delegates to TRANSITION regime for backward compatibility.
    [[nodiscard]] std::expected<void, std::string>
    run_pipeline(
        std::span<float>       alpha,
        std::span<const float> f_vol
    ) const noexcept {
        return run_pipeline(alpha, f_vol, Regime::TRANSITION);
    }

    [[nodiscard]] float vol_target()  const noexcept { return vol_target_; }
    [[nodiscard]] float weight_cap()  const noexcept { return weight_cap_; }

private:
    float vol_target_;
    float weight_cap_;
};

} // namespace alpha_pod
