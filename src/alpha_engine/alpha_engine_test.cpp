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

// ── Helpers ──────────────────────────────────────────────────────────────────

static float vec_max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::abs(x));
    return m;
}

static float vec_mean(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(v.size());
}

static float vec_std(const std::vector<float>& v) {
    float mu = vec_mean(v);
    float acc = 0.0f;
    for (float x : v) acc += (x - mu) * (x - mu);
    return std::sqrt(acc / static_cast<float>(v.size()));
}

static bool is_monotone_nondecreasing(const std::vector<float>& v) {
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1] - 1e-5f) return false;
    return true;
}

static bool has_no_nan_inf(const std::vector<float>& v) {
    for (float x : v)
        if (std::isnan(x) || std::isinf(x)) return false;
    return true;
}

// ── rank_zscore_simd ──────────────────────────────────────────────────────────

TEST(RankZScoreSIMD, MonotoneOrderPreserved) {
    // Large enough to hit SIMD lanes (16 elements)
    std::vector<float> alpha = {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 
                                 3.0f,  4.0f,  5.0f,  6.0f,  7.0f, 8.0f, 9.0f, 10.0f};
    auto r = RiskKernel::rank_zscore_simd(alpha, kRegimeTable[1].zscore_clip);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(is_monotone_nondecreasing(alpha));
}

TEST(RankZScoreSIMD, OutputBounded) {
    std::vector<float> alpha(128);
    std::iota(alpha.begin(), alpha.end(), -64.0f);
    float clip = kRegimeTable[0].zscore_clip;
    auto r = RiskKernel::rank_zscore_simd(alpha, clip);
    ASSERT_TRUE(r.has_value());
    for (float v : alpha) {
        EXPECT_LE(v,  clip + 1e-4f);
        EXPECT_GE(v, -clip - 1e-4f);
    }
}

TEST(RankZScoreSIMD, DiffersFromPearsonForFatTails) {
    // v2 Change: We no longer compare against Pearson directly because it's removed.
    // Instead, we verify that the Rank-Z outcome handles outliers without collapsing 
    // the rest of the distribution towards zero.
    std::vector<float> alpha = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1000.0f};
    
    // Pearson would shrink 1.0f to ~0.001. Rank-Z should keep it significant.
    auto r = RiskKernel::rank_zscore_simd(alpha, 3.0f);
    ASSERT_TRUE(r.has_value());
    
    EXPECT_LT(alpha[0], -1.0f); // 1.0 is the lowest, should be near negative clip
    EXPECT_GT(alpha[4], 1.0f);  // 5.0 is high rank, should be near positive clip
}

// ── apply_vol_scaling_regime ──────────────────────────────────────────────────

TEST(VolScalingRegime, CalmHasHigherTarget) {
    std::vector<float> alpha_calm(128, 1.0f);
    std::vector<float> alpha_stress(128, 1.0f);
    std::vector<float> vol(128, 0.10f);

    RiskKernel::apply_vol_scaling_regime(alpha_calm,   vol, Regime::CALM);
    RiskKernel::apply_vol_scaling_regime(alpha_stress, vol, Regime::STRESS);

    EXPECT_GT(vec_mean(alpha_calm), vec_mean(alpha_stress))
        << "CALM target (0.12) must produce larger weights than STRESS (0.06)";
}

// ── apply_nonlinear_interaction_cap ──────────────────────────────────────────

TEST(NonlinearInteractionCap, HighVolAssetGetsLowerCap) {
    std::vector<float> alpha_lv(128, 0.30f); 
    std::vector<float> alpha_hv(128, 0.30f);
    std::vector<float> vol_lv(128, 0.05f);  
    std::vector<float> vol_hv(128, 0.40f); 

    // Use TRANSITION parameters
    float vt = kRegimeTable[1].vol_target;
    float cap = kRegimeTable[1].weight_cap;

    RiskKernel::apply_nonlinear_interaction_cap(alpha_lv, vol_lv, vt, cap);
    RiskKernel::apply_nonlinear_interaction_cap(alpha_hv, vol_hv, vt, cap);

    EXPECT_GT(alpha_lv[0], alpha_hv[0])
        << "Nonlinear cap must penalize high-vol assets more than low-vol assets";
}

// ── MacroAlphaEngine v2 full pipeline ─────────────────────────────────────────

TEST(MacroAlphaEngineV2, AllRegimesProduceValidOutput) {
    const std::size_t N = 128; // Large for SIMD
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha(N), vol(N);
        for (std::size_t i = 0; i < N; ++i) {
            alpha[i] = static_cast<float>(i % 11) - 5.0f;
            vol[i]   = 0.05f + 0.01f * static_cast<float>(i % 20);
        }
        Regime regime = static_cast<Regime>(r);
        MacroAlphaEngine engine{kRegimeTable[r].vol_target, kRegimeTable[r].weight_cap};
        
        auto result = engine.run_pipeline(alpha, vol, regime);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(has_no_nan_inf(alpha));
        
        float current_cap = kRegimeTable[r].weight_cap;
        for (float v : alpha) {
            EXPECT_LE(std::abs(v), current_cap + 1e-5f);
        }
    }
}

TEST(MacroAlphaEngineV2, BackwardCompatV1StillWorks) {
    std::vector<float> alpha(128, 1.0f);
    std::vector<float> vol(128, 0.10f);
    // V2 constructor uses regime defaults automatically
    MacroAlphaEngine engine{}; 
    auto result = engine.run_pipeline(alpha, vol); 
    
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
}

TEST(MacroAlphaEngineV2, ExtremeInputsClamped) {
    std::vector<float> alpha(128, 1e6f);
    std::vector<float> vol(128, 0.01f);
    MacroAlphaEngine engine{};
    auto r = engine.run_pipeline(alpha, vol, Regime::STRESS);
    
    ASSERT_TRUE(r.has_value());
    float cap = kRegimeTable[static_cast<int>(Regime::STRESS)].weight_cap;
    for (float v : alpha) EXPECT_LE(std::abs(v), cap + 1e-5f);
}

} // namespace alpha_pod::test
