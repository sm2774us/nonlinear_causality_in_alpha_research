// src/alpha_engine/RiskKernel.h
#pragma once
#include <span>
#include <string>
#include <expected>

namespace alpha_pod {
    enum class Regime : int { CALM = 0, TRANSITION = 1, STRESS = 2 };

    struct RiskKernel {
        static std::expected<void, std::string> rank_zscore_simd(std::span<float> alpha, float clip) noexcept;
        static std::expected<void, std::string> apply_vol_scaling_regime(std::span<float> alpha, std::span<const float> f_vol, Regime regime) noexcept;
        static std::expected<void, std::string> apply_nonlinear_interaction_cap(std::span<float> alpha, std::span<const float> f_vol, float vt, float cap) noexcept;
        static std::expected<void, std::string> apply_circuit_breaker_regime(std::span<float> alpha, Regime regime) noexcept;
    };
}