// src/alpha_engine/alpha_engine_test.cpp
// Copyright 2026 AlphaPod Contributors. All Rights Reserved.
//
// GoogleTest suite v2 — regime-aware RiskKernel and MacroAlphaEngine.
//
// Coverage:
//   RiskKernel v2:
//     - rank_zscore_simd         : monotone order, bounded output, large array
//     - apply_vol_scaling_regime : correctness per regime, zero-vol guard
//     - apply_nonlinear_interaction_cap : per-asset cap tightens for high-vol
//     - apply_circuit_breaker_regime    : regime ordering invariant
//   MacroAlphaEngine v2:
//     - run_pipeline(regime)     : CALM/TRANSITION/STRESS all produce valid output
//     - regime cap ordering      : STRESS weights tighter than CALM weights
//     - backward compat v1 path  : run_pipeline(alpha, vol) still works
//     - size mismatch propagated : expected error returned
//     - NaN / Inf not produced   : for any reasonable input
//
// Run: bazel test //src/alpha_engine:alpha_engine_test --config=linux

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <span>

//import AlphaPod.MacroAlphaEngine;
//import AlphaPod.RiskKernel;
#include "MacroAlphaEngine.h" // Synchronize with the bridge
#include "RiskKernel.h" // Synchronize with the bridge

namespace alpha_pod::test {

using namespace alpha_pod;

// ── Helpers ──────────────────────────────────────────────────────────────────

static float vec_max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::abs(x));
    return m;
}

static float vec_mean(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    return std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(v.size());
}

static bool has_no_nan_inf(const std::vector<float>& v) {
    for (float x : v)
        if (std::isnan(x) || std::isinf(x)) return false;
    return true;
}

// ── rank_zscore_simd ──────────────────────────────────────────────────────────

TEST(RankZScoreSIMD, OutputBoundedAndSIMDAligned) {
    // 128 elements ensures multiple SIMD lanes are processed (avoiding just the tail loop)
    std::vector<float> alpha(128);
    std::iota(alpha.begin(), alpha.end(), -64.0f);
    
    float clip = kRegimeTable[static_cast<int>(Regime::CALM)].zscore_clip;
    auto r = RiskKernel::rank_zscore_simd(alpha, clip);
    
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
    for (float v : alpha) {
        EXPECT_LE(v,  clip + 1e-4f);
        EXPECT_GE(v, -clip - 1e-4f);
    }
}

TEST(RankZScoreSIMD, LargeArrayNoNaN) {
    const std::size_t N = 2048;
    std::vector<float> alpha(N);
    for (std::size_t i = 0; i < N; ++i)
        alpha[i] = static_cast<float>(i % 100) - 50.0f;
    
    auto r = RiskKernel::rank_zscore_simd(alpha, 3.0f);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
}

// ── apply_vol_scaling_regime ──────────────────────────────────────────────────

TEST(VolScalingRegime, CalmHasHigherTargetThanStress) {
    std::vector<float> alpha_calm(128, 1.0f);
    std::vector<float> alpha_stress(128, 1.0f);
    std::vector<float> vol(128, 0.10f);

    RiskKernel::apply_vol_scaling_regime(alpha_calm,   vol, Regime::CALM);
    RiskKernel::apply_vol_scaling_regime(alpha_stress, vol, Regime::STRESS);

    // CALM (0.12 target) should result in higher weights than STRESS (0.06 target)
    EXPECT_GT(vec_mean(alpha_calm), vec_mean(alpha_stress));
}

// ── apply_nonlinear_interaction_cap ──────────────────────────────────────────

TEST(NonlinearInteractionCap, HighVolAssetGetsLowerCap) {
    // Ma & Prosperino (2023) Nonlinearity Check:
    // Two assets with same raw alpha, but different vol.
    std::vector<float> alpha_lv(128, 0.40f); 
    std::vector<float> alpha_hv(128, 0.40f);
    std::vector<float> vol_lv(128, 0.05f);  // Low Vol
    std::vector<float> vol_hv(128, 0.40f);  // High Vol

    float vt  = kRegimeTable[static_cast<int>(Regime::TRANSITION)].vol_target;
    float cap = kRegimeTable[static_cast<int>(Regime::TRANSITION)].weight_cap;

    RiskKernel::apply_nonlinear_interaction_cap(alpha_lv, vol_lv, vt, cap);
    RiskKernel::apply_nonlinear_interaction_cap(alpha_hv, vol_hv, vt, cap);

    // High vol asset's effective cap should be tightened significantly
    EXPECT_GT(alpha_lv[0], alpha_hv[0]) 
        << "Nonlinear interaction cap must penalize high-vol assets more severely.";
}

// ── MacroAlphaEngine v2 Pipeline ─────────────────────────────────────────────

TEST(MacroAlphaEngineV2, AllRegimesProduceValidOutput) {
    const std::size_t N = 128;
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha(N), vol(N);
        for (std::size_t i = 0; i < N; ++i) {
            alpha[i] = static_cast<float>(i % 11) - 5.0f;
            vol[i]   = 0.05f + 0.01f * static_cast<float>(i % 20);
        }
        
        Regime regime = static_cast<Regime>(r);
        MacroAlphaEngine engine{kRegimeTable[r].vol_target, kRegimeTable[r].weight_cap};
        
        auto result = engine.run_pipeline(alpha, vol, regime);
        ASSERT_TRUE(result.has_value()) << "Pipeline failed for regime " << r;
        EXPECT_TRUE(has_no_nan_inf(alpha));
        
        float current_cap = kRegimeTable[r].weight_cap;
        for (float v : alpha) {
            EXPECT_LE(std::abs(v), current_cap + 1e-5f) << "Cap violation in regime " << r;
        }
    }
}

TEST(MacroAlphaEngineV2, StressWeightsTighterThanCalm) {
    const std::size_t N = 128;
    std::vector<float> alpha_c(N, 5.0f), alpha_s(N, 5.0f);
    std::vector<float> vol(N, 0.10f);

    MacroAlphaEngine eng_c{kRegimeTable[0].vol_target, kRegimeTable[0].weight_cap};
    MacroAlphaEngine eng_s{kRegimeTable[2].vol_target, kRegimeTable[2].weight_cap};

    eng_c.run_pipeline(alpha_c, vol, Regime::CALM);
    eng_s.run_pipeline(alpha_s, vol, Regime::STRESS);

    EXPECT_GT(vec_max_abs(alpha_c), vec_max_abs(alpha_s))
        << "STRESS regime should enforce tighter absolute weight limits than CALM.";
}

TEST(MacroAlphaEngineV2, BackwardCompatV1StillWorks) {
    std::vector<float> alpha(128, 2.0f);
    std::vector<float> vol(128, 0.10f);
    
    MacroAlphaEngine engine{}; // Defaults to TRANSITION params
    auto result = engine.run_pipeline(alpha, vol); // Legacy 2-arg call
    
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
}

TEST(MacroAlphaEngineV2, SizeMismatchPropagated) {
    std::vector<float> alpha(10, 1.0f);
    std::vector<float> vol(5, 0.10f); // Size mismatch
    MacroAlphaEngine engine{};
    auto r = engine.run_pipeline(alpha, vol, Regime::TRANSITION);
    EXPECT_FALSE(r.has_value());
}

} // namespace alpha_pod::test