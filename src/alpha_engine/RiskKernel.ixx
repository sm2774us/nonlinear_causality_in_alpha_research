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
module; // Start of Global Module Fragment

// import std;
// import std.simd;

// Standard Library Headers for C++26 compatibility in Bazel Sandbox
#include <vector>        // For std::vector
#include <numeric>       // For std::iota
#include <algorithm>     // For std::sort, std::clamp
#include <array>         // For std::array
#include <span>          // For std::span
#include <expected>      // For std::expected
#include <string>        // For std::string error messages
#include <cstddef>       // For std::size_t

// SIMD Header (P1928 / C++26 experimental support)
#include <experimental/simd>

export module AlphaPod.RiskKernel;

using namespace std::experimental;

namespace alpha_pod {

// ── Regime parameter tables (compile-time) ────────────────────────────────────

/// Market regime identifiers (must match Python RegimeState enum).
export enum class Regime : int {
    CALM       = 0,
    TRANSITION = 1,
    STRESS     = 2,
};

export struct RegimeParams {
    float vol_target;
    float weight_cap;
    float zscore_clip;
};

/// Compile-time regime parameter table.
/// Source: calibrated against Ma & Prosperino (2023) regime characterisation.
export inline constexpr std::array<RegimeParams, 3> kRegimeTable{{
    {0.12f, 0.25f, 3.0f},   // CALM:       more aggressive, wider clip
    {0.10f, 0.20f, 3.0f},   // TRANSITION: baseline
    {0.06f, 0.12f, 2.0f},   // STRESS:     de-risk; tighter clip (fat-tail guard)
}};

// ── Constants ─────────────────────────────────────────────────────────────────

inline constexpr float kVolEps     = 1e-6f;
inline constexpr float kDefaultCap = 0.20f;

// ── RiskKernel v2 ─────────────────────────────────────────────────────────────

/// @brief Stateless SIMD risk transformations — regime-conditional nonlinear kernel.
export class RiskKernel {
public:

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
    [[nodiscard]] static std::expected<void, std::string>
    rank_zscore_simd(std::span<float> alpha, float clip = 3.0f) noexcept {
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
        using simd_t = native_simd<float>;
        const std::size_t step = simd_t::size();
        const float scale = (N > 1) ? (2.0f * clip / static_cast<float>(N - 1)) : 0.0f;
        const simd_t s_v  = scale;
        const simd_t o_v  = clip;
        const simd_t lo   = -clip;
        const simd_t hi   =  clip;

        // std::size_t i = 0;
        // for (; i + step <= N; i += step) {
        //     simd_t r(ranks.data() + i, element_aligned);
        //
        //     // Process: The standard-compliant way to clamp in SIMD TS
        //     simd_t product = r * s_v;
        //     simd_t val = product - o_v;
        //
        //     // Start with the calculated value
        //     simd_t clamped_v = val;
        //     // "Where the value is too low, set it to lo"
        //     where(clamped_v < lo, clamped_v) = lo;
        //
        //     // "Where the value is too high, set it to hi"
        //     where(clamped_v > hi, clamped_v) = hi;
        //
        //     clamped_v.copy_to(alpha.data() + i, element_aligned);
        // }

        std::size_t i = 0;
        for (; i + step <= N; i += step) {
            // 1. Initialize 'r' by loading data into it
            simd_t r;
            r.copy_from(ranks.data() + i, element_aligned);

            simd_t val;

            // 2. Combine Math and Clamp for efficiency
            for (std::size_t n = 0; n < step; ++n) {
                // Extract to float to avoid simd_reference errors
                float r_lane = r[n];

                // Perform math
                float calculated = (r_lane * scale) - clip;

                // Clamp (now types match perfectly: float, float, float)
                val[n] = std::clamp(calculated, -clip, clip);
            }

            // 3. Store results
            val.copy_to(alpha.data() + i, element_aligned);
        }
        for (; i < N; ++i)
            alpha[i] = std::clamp(ranks[i] * scale - clip, -clip, clip);

        // // rank_zscore_simd: replace the simd block with:
        // const float scale = (N > 1) ? (2.0f * clip / static_cast<float>(N - 1)) : 0.0f;
        // for (std::size_t i = 0; i < N; ++i)
        //     alpha[i] = std::clamp(ranks[i] * scale - clip, -clip, clip);

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
    [[nodiscard]] static std::expected<void, std::string>
    apply_vol_scaling_regime(
        std::span<float>       alpha,
        std::span<const float> f_vol,
        Regime                 regime
    ) noexcept {
        if (alpha.size() != f_vol.size()) [[unlikely]]
            return std::unexpected{"apply_vol_scaling_regime: size mismatch"};

        const float vt = kRegimeTable[static_cast<int>(regime)].vol_target;

        using simd_t = native_simd<float>;
        const std::size_t N    = alpha.size();
        const std::size_t step = simd_t::size();
        const simd_t vt_v = vt;
        const simd_t eps  = kVolEps;

        std::size_t i = 0;
        for (; i + step <= N; i += step) {
            // 1. Load data into SIMD registers
            simd_t a_vec;
            a_vec.copy_from(alpha.data() + i, element_aligned);

            simd_t v_vec;
            v_vec.copy_from(f_vol.data() + i, element_aligned);

            simd_t result;

            // 2. Perform the math per-lane
            // Clang 19 will optimize this back into pure SIMD instructions
            for (std::size_t n = 0; n < step; ++n) {
                // Calculation: a * (vt / (v + eps))
                result[n] = a_vec[n] * (vt / (v_vec[n] + kVolEps));
            }

            // 3. Store back to memory
            result.copy_to(alpha.data() + i, element_aligned);
        }

        // Scalar Tail
        for (; i < N; ++i)
            alpha[i] *= vt / (f_vol[i] + kVolEps);

        // // apply_vol_scaling_regime: replace simd block with:
        // const std::size_t N = alpha.size();
        // for (std::size_t i = 0; i < N; ++i)
        //     alpha[i] *= vt / (f_vol[i] + kVolEps);

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
    [[nodiscard]] static std::expected<void, std::string>
    apply_circuit_breaker_regime(
        std::span<float> alpha,
        Regime           regime
    ) noexcept {
        const float cap = kRegimeTable[static_cast<int>(regime)].weight_cap;
        if (cap <= 0.0f) [[unlikely]]
            return std::unexpected{"apply_circuit_breaker_regime: cap must be > 0"};

        using simd_t = native_simd<float>;
        const std::size_t N    = alpha.size();
        const std::size_t step = simd_t::size();

        // Use float scalars for the clamp boundaries
        const float lo = -cap;
        const float hi =  cap;

        std::size_t i = 0;
        for (; i + step <= N; i += step) {
            // 1. Load data into the SIMD register
            simd_t a;
            a.copy_from(alpha.data() + i, element_aligned);

            // 2. Create a result register
            simd_t result;

            // 3. Process each lane using member operator[]
            // This bypasses the hidden stdx::clamp non-member function
            for (std::size_t n = 0; n < step; ++n) {
                // Extract to float to avoid simd_reference template issues
                float val = a[n];
                result[n] = std::clamp(val, lo, hi);
            }

            // 4. Store back to memory
            result.copy_to(alpha.data() + i, element_aligned);
        }
        for (; i < N; ++i)
            alpha[i] = std::clamp(alpha[i], -cap, cap);

        // // apply_circuit_breaker_regime: replace simd block with:
        // const std::size_t N = alpha.size();
        // for (std::size_t i = 0; i < N; ++i)
        //     alpha[i] = std::clamp(alpha[i], -cap, cap);

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
    [[nodiscard]] static std::expected<void, std::string>
    apply_nonlinear_interaction_cap(
        std::span<float>       alpha,
        std::span<const float> f_vol,
        float                  vol_target,
        float                  base_cap
    ) noexcept {
        if (alpha.size() != f_vol.size()) [[unlikely]]
            return std::unexpected{"apply_nonlinear_interaction_cap: size mismatch"};

        using simd_t = native_simd<float>;
        const std::size_t N    = alpha.size();
        const std::size_t step = simd_t::size();
        const simd_t vt   = vol_target;
        const simd_t cap  = base_cap;
        const simd_t one  = 1.0f;
        const simd_t eps  = kVolEps;

        std::size_t i = 0;
        for (; i + step <= N; i += step) {
            // 1. Load data
            simd_t a_vec;
            a_vec.copy_from(alpha.data() + i, element_aligned);

            simd_t v_vec;
            v_vec.copy_from(f_vol.data() + i, element_aligned);

            simd_t result;

            // 2. Process each lane
            for (std::size_t n = 0; n < step; ++n) {
                // Extract to plain float to avoid template deduction failures
                float current_alpha = a_vec[n];
                float current_vol   = v_vec[n];

                // Calculate effective_cap
                float ratio = current_vol / (vol_target + kVolEps);
                if (ratio < 1.0f) ratio = 1.0f;

                float eff_cap = base_cap / ratio;

                // std::clamp now sees (float, float, float) -> SUCCESS
                result[n] = std::clamp(current_alpha, -eff_cap, eff_cap);
            }

            // 3. Store the results back
            result.copy_to(alpha.data() + i, element_aligned);
        }
        for (; i < N; ++i) {
            float eff = base_cap / std::max(f_vol[i] / (vol_target + kVolEps), 1.0f);
            alpha[i]  = std::clamp(alpha[i], -eff, eff);
        }

        // // apply_nonlinear_interaction_cap: replace simd block with:
        // const std::size_t N = alpha.size();
        // for (std::size_t i = 0; i < N; ++i) {
        //     float eff = base_cap / std::max(f_vol[i] / (vol_target + kVolEps), 1.0f);
        //     alpha[i]  = std::clamp(alpha[i], -eff, eff);
        // }

        return {};
    }

    // ── v1 functions retained for backward compatibility ──────────────────────

    [[nodiscard]] static std::expected<void, std::string>
    apply_vol_scaling(
        std::span<float>       alpha,
        std::span<const float> f_vol,
        float                  vol_target
    ) noexcept {
        return apply_vol_scaling_regime(alpha, f_vol, Regime::TRANSITION);
    }

    [[nodiscard]] static std::expected<void, std::string>
    apply_circuit_breaker(
        std::span<float> alpha,
        float            cap = kDefaultCap
    ) noexcept {
        (void)cap;  // cap comes from regime table in v2
        return apply_circuit_breaker_regime(alpha, Regime::TRANSITION);
    }

    [[nodiscard]] static std::expected<void, std::string>
    cross_sectional_zscore(std::span<float> alpha) noexcept {
        // v2 delegates to rank-based z-score
        return rank_zscore_simd(alpha, 3.0f);
    }
};

} // namespace alpha_pod