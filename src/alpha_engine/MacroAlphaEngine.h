// src/alpha_engine/MacroAlphaEngine.h
#pragma once
#include <vector>
#include <span>
#include <string>
#include <expected>

namespace alpha_pod {
    enum class Regime : int { CALM = 0, TRANSITION = 1, STRESS = 2 };

    class MacroAlphaEngine {
    public:
        MacroAlphaEngine() = default;
        std::expected<void, std::string> run_pipeline(
            std::span<float> alpha, 
            std::span<const float> f_vol, 
            Regime regime = Regime::TRANSITION
        ) const noexcept;
    };
}