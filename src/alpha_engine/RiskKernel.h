// src/alpha_engine/RiskKernel.h
#pragma once

#include <span>
#include <string>
#include <expected>
#include <array>

namespace alpha_pod {
    enum class Regime : int { CALM = 0, TRANSITION = 1, STRESS = 2 };

    struct RegimeParams {
        float vol_target;
        float weight_cap;
        float zscore_clip;
    };

    // Aligned with Ma & Prosperino (2023)
    inline constexpr std::array<RegimeParams, 3> kRegimeTable{{
        {0.12f, 0.25f, 3.0f},   // CALM
        {0.10f, 0.20f, 3.0f},   // TRANSITION
        {0.06f, 0.12f, 2.0f},   // STRESS
    }};

    struct RiskKernel {
        [[nodiscard]] static std::expected<void, std::string> rank_zscore_simd(std::span<float> alpha, float clip) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_vol_scaling_regime(std::span<float> alpha, std::span<const float> f_vol, Regime regime) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_nonlinear_interaction_cap(std::span<float> alpha, std::span<const float> f_vol, float vt, float cap) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_circuit_breaker_regime(std::span<float> alpha, Regime regime) noexcept;
        
        [[nodiscard]] static std::expected<void, std::string> apply_vol_scaling(std::span<float> alpha, std::span<const float> f_vol, float vol_target) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_circuit_breaker(std::span<float> alpha, float cap) noexcept;
    };
}