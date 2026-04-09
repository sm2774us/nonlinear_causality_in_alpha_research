// src/alpha_engine/RiskKernel.h
#pragma once
#include <span>
#include <string>
#include <expected>

namespace alpha_pod {
    enum class Regime : int { CALM = 0, TRANSITION = 1, STRESS = 2 };

    struct RegimeParams {
        float vol_target;
        float weight_cap;
        float zscore_clip;
    };

    // Forward declare as a raw array.
    // Forward declare: Definition moves to RiskKernel.ixx
    // This is 100% ABI compatible and requires ZERO standard library headers.
    extern const RegimeParams kRegimeTable[3];

    struct RiskKernel {
        [[nodiscard]] static std::expected<void, std::string> rank_zscore_simd(std::span<float> alpha, float clip) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_vol_scaling_regime(std::span<float> alpha, std::span<const float> f_vol, Regime regime) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_nonlinear_interaction_cap(std::span<float> alpha, std::span<const float> f_vol, float vt, float cap) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_circuit_breaker_regime(std::span<float> alpha, Regime regime) noexcept;
        
        [[nodiscard]] static std::expected<void, std::string> apply_vol_scaling(std::span<float> alpha, std::span<const float> f_vol, float vol_target) noexcept;
        [[nodiscard]] static std::expected<void, std::string> apply_circuit_breaker(std::span<float> alpha, float cap) noexcept;
    };
}