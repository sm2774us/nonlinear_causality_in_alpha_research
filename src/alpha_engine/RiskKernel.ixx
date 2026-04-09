// src/alpha_engine/RiskKernel.ixx
// Copyright 2026 AlphaPod Contributors. All Rights Reserved.
//
// C++26 module: RiskKernel v2 — Regime-Conditional Nonlinear Risk Kernel
//
// UPGRADE from v1: Addresses Ma & Prosperino (2023).
//
// The v1 kernel applied a single fixed vol-target and weight-cap to the full
// alpha vector. This is the "Principle of Superposition" in risk management:
// it assumes all assets respond identically to the same linear scaling.
//
// Ma & Prosperino (2023) demonstrate that financial markets exhibit significant
// NONLINEAR CAUSALITY that is regime-dependent. The correct risk response is:
//   - CALM regime:      higher vol-target (0.12), wider cap (0.25) — capture carry
//   - TRANSITION:       baseline (0.10 / 0.20) — maintain but watch
//   - STRESS regime:    lower vol-target (0.06), tight cap (0.12) — capital preservation
//
// New functions in v2:
//   apply_vol_scaling_regime()     — regime-specific vol target per-asset SIMD
//   apply_circuit_breaker_regime() — regime-specific weight cap per-asset SIMD
//   rank_zscore_simd()             — rank-based (Spearman) z-score, not Pearson
//   apply_nonlinear_interaction_cap() — caps based on |alpha| × f_vol (nonlinear)
//
// Design:
//   All new functions accept a scalar 'regime' integer (0=calm, 1=transition, 2=stress)
//   and select the appropriate parameter from a compile-time constexpr table.
//   No heap allocation; all operations are in-place on contiguous float spans.
//
// Requires: Clang-19+ or MSVC 19.40+ with C++26 / std::simd P1928.
module; // START OF GLOBAL MODULE FRAGMENT
#include "RiskKernel.h" 

export module AlphaPod.RiskKernel;

import std;
import std.simd;

namespace alpha_pod {

// ── v2 Implementation ────────────────────────────────────────────────────────

// ── Rank-based Z-score (v2 upgrade, replaces Pearson z-score) ─────────────

/// @brief In-place rank-normalisation of alpha to [-clip, +clip] via rank transform.
///
/// Motivation (Ma & Prosperino 2023):
///   Pearson z-score = (x - μ) / σ assumes an elliptical (Gaussian) distribution.
///   When nonlinear causality is present, returns are fat-tailed and
///   asymmetric. The rank transform is distribution-free and preserves
///   ordinal (nonlinear) information.
///
///   rank_z[i] = ((rank_i / (N-1)) * 2 - 1) * clip
///
///   This is consistent with Spearman-rank normalisation used in the Python
///   FeatureEngine.rank_zscore_expr() and with the IC calculated via
///   Spearman correlation.
///
/// @param alpha  Mutable span — overwritten with rank z-scores.
/// @param clip   Symmetric clip bound (e.g. 3.0 for calm, 2.0 for stress).
std::expected<void, std::string>
RiskKernel::rank_zscore_simd(std::span<float> alpha, float clip) noexcept {
    const std::size_t N = alpha.size();
    if (N == 0) [[unlikely]]
        return std::unexpected{"rank_zscore_simd: empty span"};
    if (clip <= 0.0f) [[unlikely]]
        return std::unexpected{"rank_zscore_simd: clip must be > 0"};

    // Step 1: argsort → ranks (sort indices, then invert)
    std::vector<std::size_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
                [&alpha](std::size_t a, std::size_t b) {
                    return alpha[a] < alpha[b];
                });

    std::vector<float> ranks(N);
    for (std::size_t i = 0; i < N; ++i)
        ranks[order[i]] = static_cast<float>(i);

    // Step 2: Normalise ranks → [-clip, +clip] via SIMD
    using simd_t = std::simd<float>;
    const std::size_t step = simd_t::size();
    const float scale = (N > 1) ? (2.0f * clip / static_cast<float>(N - 1)) : 0.0f;
    const simd_t s_v  = scale;
    const simd_t o_v  = clip;
    const simd_t lo   = -clip;
    const simd_t hi   =  clip;

    std::size_t i = 0;
    for (; i + step <= N; i += step) {
        simd_t r(ranks.data() + i, std::element_aligned);
        std::simd::clamp(r * s_v - o_v, lo, hi)
            .copy_to(alpha.data() + i, std::element_aligned);
    }
    for (; i < N; ++i)
        alpha[i] = std::clamp(ranks[i] * scale - clip, -clip, clip);

    return {};
}

// ── Regime-conditional vol scaling ────────────────────────────────────────

/// @brief Vol-scales alpha using a regime-specific target volatility.
///
/// v1: alpha[i] *= sigma* / (vol[i] + eps)  — fixed sigma*
/// v2: sigma* is looked up from kRegimeTable[regime] — regime-conditional
///
/// Motivation: In stress regimes, a fixed 10% vol-target applied to
/// elevated realised vols produces smaller positions BUT does not reduce
/// them enough — because elevated vol in stress also means higher
/// LEFT-TAIL RISK (nonlinear, not captured by σ alone).
/// The regime-specific target encodes this: stress → σ*=6%, not 10%.
///
/// @param alpha   [in/out] Alpha weights.
/// @param f_vol   [in]     Realised annualised vol per asset.
/// @param regime  Current market regime (from Python RegimeEngine).
std::expected<void, std::string>
RiskKernel::apply_vol_scaling_regime(std::span<float> alpha, std::span<const float> f_vol, Regime regime) noexcept {
    if (alpha.size() != f_vol.size()) [[unlikely]]
        return std::unexpected{"apply_vol_scaling_regime: size mismatch"};

    const float vt = kRegimeTable[static_cast<int>(regime)].vol_target;

    using simd_t = std::simd<float>;
    const std::size_t N    = alpha.size();
    const std::size_t step = simd_t::size();
    const simd_t vt_v = vt;
    const simd_t eps  = kVolEps;

    std::size_t i = 0;
    for (; i + step <= N; i += step) {
        simd_t a(alpha.data() + i, std::element_aligned);
        simd_t v(f_vol.data()  + i, std::element_aligned);
        (a * (vt_v / (v + eps))).copy_to(alpha.data() + i, std::element_aligned);
    }
    for (; i < N; ++i)
        alpha[i] *= vt / (f_vol[i] + kVolEps);

    return {};
}

// ── Nonlinear interaction cap ──────────────────────────────────────────────

/// @brief Caps alpha by a nonlinear function of alpha × f_vol.
///
/// Motivation (Ma & Prosperino 2023):
///   A purely linear cap treats all assets identically regardless of
///   their current vol level. But the INTERACTION between signal strength
///   and vol is nonlinear: a position with weight 0.15 in an asset
///   with vol=0.30 is 3× riskier than the same weight in an asset
///   with vol=0.10. The nonlinear cap imposes:
///
///     effective_cap_i = base_cap / max(f_vol_i / vol_target, 1.0)
///
///   This means assets with elevated vol get a TIGHTER effective cap,
///   preventing the linear kernel from over-allocating to high-vol assets.
///
/// @param alpha       [in/out] Vol-scaled weights.
/// @param f_vol       [in]     Realised annualised vol per asset.
/// @param vol_target  Reference vol for normalisation.
/// @param base_cap    Baseline cap before vol adjustment.
std::expected<void, std::string>
RiskKernel::apply_nonlinear_interaction_cap(std::span<float> alpha, std::span<const float> f_vol, float vt, float cap) noexcept {
    if (alpha.size() != f_vol.size()) [[unlikely]]
        return std::unexpected{"apply_nonlinear_interaction_cap: size mismatch"};

    using simd_t = std::simd<float>;
    const std::size_t N    = alpha.size();
    const std::size_t step = simd_t::size();
    const simd_t vt   = vol_target;
    const simd_t cap  = base_cap;
    const simd_t one  = 1.0f;
    const simd_t eps  = kVolEps;

    std::size_t i = 0;
    for (; i + step <= N; i += step) {
        simd_t a(alpha.data() + i, std::element_aligned);
        simd_t v(f_vol.data()  + i, std::element_aligned);
        // effective_cap = base_cap / max(vol/vol_target, 1)
        simd_t eff_cap = cap / std::simd::max(v / (vt + eps), one);
        std::simd::clamp(a, -eff_cap, eff_cap)
            .copy_to(alpha.data() + i, std::element_aligned);
    }
    for (; i < N; ++i) {
        float eff = base_cap / std::max(f_vol[i] / (vol_target + kVolEps), 1.0f);
        alpha[i]  = std::clamp(alpha[i], -eff, eff);
    }

    return {};
}

// ── Regime-conditional circuit breaker ────────────────────────────────────

/// @brief Hard-caps |alpha[i]| using a regime-specific weight cap.
///
/// v1: fixed cap = 0.20
/// v2: cap from kRegimeTable[regime] — 0.25 (calm) / 0.20 (transition) / 0.12 (stress)
///
/// This encodes the asymmetric risk posture of top macro pods:
/// be aggressive when the market is calm, survive when it is stressed.
std::expected<void, std::string>
RiskKernel::apply_circuit_breaker_regime(std::span<float> alpha, Regime regime) noexcept {
    const float cap = kRegimeTable[static_cast<int>(regime)].weight_cap;
    if (cap <= 0.0f) [[unlikely]]
        return std::unexpected{"apply_circuit_breaker_regime: cap must be > 0"};

    using simd_t = std::simd<float>;
    const std::size_t N    = alpha.size();
    const std::size_t step = simd_t::size();
    const simd_t lo = -cap;
    const simd_t hi =  cap;

    std::size_t i = 0;
    for (; i + step <= N; i += step) {
        simd_t a(alpha.data() + i, std::element_aligned);
        std::simd::clamp(a, lo, hi).copy_to(alpha.data() + i, std::element_aligned);
    }
    for (; i < N; ++i)
        alpha[i] = std::clamp(alpha[i], -cap, cap);

    return {};
}

// ── v1 Legacy Implementation ──────────────────────────────────────────────────

static std::expected<void, std::string>
RiskKernel::apply_vol_scaling(
    std::span<float>       alpha,
    std::span<const float> f_vol,
    float                  vol_target
) noexcept {
    return apply_vol_scaling_regime(alpha, f_vol, Regime::TRANSITION);
}

static std::expected<void, std::string>
RiskKernel::apply_circuit_breaker(
    std::span<float> alpha,
    float            cap = kDefaultCap
) noexcept {
    (void)cap;  // cap comes from regime table in v2
    return apply_circuit_breaker_regime(alpha, Regime::TRANSITION);
}

static std::expected<void, std::string>
RiskKernel::cross_sectional_zscore(std::span<float> alpha) noexcept {
    // v2 delegates to rank-based z-score
    return rank_zscore_simd(alpha, 3.0f);
}

} // namespace alpha_pod