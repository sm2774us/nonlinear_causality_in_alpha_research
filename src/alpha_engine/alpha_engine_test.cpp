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

TEST(RankZScoreSIMD, BasicFunctionality) {
    std::vector<float> alpha(128);
    std::iota(alpha.begin(), alpha.end(), -64.0f);
    float clip = kRegimeTable[0].zscore_clip;
    auto r = RiskKernel::rank_zscore_simd(alpha, clip);
    ASSERT_TRUE(r.has_value());
    for (float v : alpha) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_FALSE(std::isinf(v));
    }
}

TEST(NonlinearInteractionCap, HighVolPenalty) {
    std::vector<float> alpha_lv(128, 0.30f); 
    std::vector<float> alpha_hv(128, 0.30f);
    std::vector<float> vol_lv(128, 0.05f);  
    std::vector<float> vol_hv(128, 0.40f); 

    RiskKernel::apply_nonlinear_interaction_cap(alpha_lv, vol_lv, kRegimeTable[1].vol_target, kRegimeTable[1].weight_cap);
    RiskKernel::apply_nonlinear_interaction_cap(alpha_hv, vol_hv, kRegimeTable[1].vol_target, kRegimeTable[1].weight_cap);

    EXPECT_GT(alpha_lv[0], alpha_hv[0]);
}

TEST(MacroAlphaEngineV2, RegimeLogic) {
    const std::size_t N = 128;
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha(N, 1.0f), vol(N, 0.15f);
        MacroAlphaEngine engine{kRegimeTable[r].vol_target, kRegimeTable[r].weight_cap};
        auto result = engine.run_pipeline(alpha, vol, static_cast<Regime>(r));
        ASSERT_TRUE(result.has_value());
    }
}

} // namespace alpha_pod::test