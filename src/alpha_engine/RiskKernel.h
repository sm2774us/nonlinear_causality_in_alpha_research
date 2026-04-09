// src/alpha_engine/RiskKernel.h
#pragma once
#include <span>
#include <string>
#include <expected>

namespace alpha_pod {

    // ── Regime parameter tables (compile-time) ────────────────────────────────────

    /// Market regime identifiers (must match Python RegimeState enum).
    export enum class Regime : int {
        CALM       = 0,
        TRANSITION = 1,
        STRESS     = 2,
    };

    struct RegimeParams {
        float vol_target;
        float weight_cap;
        float zscore_clip;
    };

    /// Compile-time regime parameter table.
    /// Source: calibrated against Ma & Prosperino (2023) regime characterisation.
    inline constexpr std::array<RegimeParams, 3> kRegimeTable{{
        {0.12f, 0.25f, 3.0f},   // CALM:       more aggressive, wider clip
        {0.10f, 0.20f, 3.0f},   // TRANSITION: baseline
        {0.06f, 0.12f, 2.0f},   // STRESS:     de-risk; tighter clip (fat-tail guard)
    }};

    // ── Constants ─────────────────────────────────────────────────────────────────

    inline constexpr float kVolEps     = 1e-6f;
    inline constexpr float kDefaultCap = 0.20f;    

    struct RiskKernel {
        static std::expected<void, std::string> rank_zscore_simd(std::span<float> alpha, float clip) noexcept;
        static std::expected<void, std::string> apply_vol_scaling_regime(std::span<float> alpha, std::span<const float> f_vol, Regime regime) noexcept;
        static std::expected<void, std::string> apply_nonlinear_interaction_cap(std::span<float> alpha, std::span<const float> f_vol, float vt, float cap) noexcept;
        static std::expected<void, std::string> apply_circuit_breaker_regime(std::span<float> alpha, Regime regime) noexcept;
    };
}