// src/alpha_engine/MacroAlphaEngine.h
#pragma once
#include "RiskKernel.h" // Necessary for kRegimeTable and Regime
#include <vector>
#include <span>
#include <string>
#include <expected> // <--- Ensure this is present

namespace alpha_pod {
    class MacroAlphaEngine {
    public:
        // Default arguments must be in the header
        MacroAlphaEngine(
            float vol_target = kRegimeTable[1].vol_target,
            float weight_cap = kRegimeTable[1].weight_cap
        );

        // Explicit v2: Requires 3 arguments
        std::expected<void, std::string> run_pipeline(
            std::span<float> alpha, 
            std::span<const float> f_vol, 
            Regime regime
        ) const noexcept;

        // Legacy v1: Requires 2 arguments
        std::expected<void, std::string> run_pipeline(
            std::span<float> alpha, 
            std::span<const float> f_vol
        ) const noexcept;

        // Added getters for nanobind def_prop_ro
        float vol_target() const noexcept;
        float weight_cap() const noexcept;

    private:
        float vol_target_;
        float weight_cap_;
    };
}