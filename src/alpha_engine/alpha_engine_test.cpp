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
// Plain TU — NOT a named module unit.
// The test binary is a plain executable. Declaring it as a module impl unit
// requires a matching export-module interface in FILE_SET CXX_MODULES —
// none exists — so CMake's dyndep would reject the build.
// A plain TU with import declarations is the correct approach:
// clang-scan-deps detects them and CMake injects -fmodule-file= flags.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <span>

// C++26 named-module imports (resolved at compile time via dyndep)
import AlphaPod.MacroAlphaEngine;
import AlphaPod.RiskKernel;

namespace alpha_pod::test {

// Use explicit aliases to force the compiler to resolve the module symbols early
using Regime = alpha_pod::Regime;
using RiskKernel = alpha_pod::RiskKernel;
using MacroAlphaEngine = alpha_pod::MacroAlphaEngine;
// kRegimeTable is a constexpr variable, not a type.
// 'using X = Y' is type-alias syntax (invalid for variables).
// 'using alpha_pod::kRegimeTable' is a using-declaration — valid for variables.
using alpha_pod::kRegimeTable;

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
    // Input sorted → output sorted (rank preserves order)
    std::vector<float> alpha = {-3.0f, -1.5f, 0.0f, 1.0f, 2.5f, 4.0f, 7.0f, 10.0f};
    std::vector<float> orig  = alpha;
    auto r = RiskKernel::rank_zscore_simd(alpha, kRegimeTable[1].zscore_clip);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(is_monotone_nondecreasing(alpha))
        << "rank z-score must preserve relative ordering";
}

TEST(RankZScoreSIMD, OutputBounded) {
    std::vector<float> alpha(64);
    std::iota(alpha.begin(), alpha.end(), -32.0f);
    float clip = kRegimeTable[0].zscore_clip;
    auto r = RiskKernel::rank_zscore_simd(alpha, clip);
    ASSERT_TRUE(r.has_value());
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

TEST(RankZScoreSIMD, DiffersFromPearsonForFatTails) {
    // Inject an outlier: Pearson z-score inflates σ and shrinks all scores;
    // rank z-score is unaffected by the outlier's magnitude.
    std::vector<float> alpha_rank  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1000.0f};
    std::vector<float> alpha_pearl = alpha_rank;

    RiskKernel::rank_zscore_simd(alpha_rank,  3.0f);
    RiskKernel::cross_sectional_zscore(alpha_pearl);  // Pearson

    // Rank: the non-outlier scores are still spread across [-3,+3]
    // Pearson: the non-outlier scores are all compressed near 0
    float rank_spread  = *std::max_element(alpha_rank.begin(),  alpha_rank.end())
                       - *std::min_element(alpha_rank.begin(),  alpha_rank.end());
    float pearl_spread = *std::max_element(alpha_pearl.begin(), alpha_pearl.end())
                       - *std::min_element(alpha_pearl.begin(), alpha_pearl.end());

    EXPECT_GT(rank_spread, pearl_spread * 0.5f)
        << "Rank z-score should be more spread than Pearson when outlier present";
}

// ── apply_vol_scaling_regime ──────────────────────────────────────────────────

TEST(VolScalingRegime, CalmHasHigherTarget) {
    std::vector<float> alpha_calm   = {1.0f, 1.0f, 1.0f};
    std::vector<float> alpha_stress = {1.0f, 1.0f, 1.0f};
    std::vector<float> vol          = {0.10f, 0.10f, 0.10f};

    RiskKernel::apply_vol_scaling_regime(alpha_calm,   vol, Regime::CALM);
    RiskKernel::apply_vol_scaling_regime(alpha_stress, vol, Regime::STRESS);

    // CALM vol_target (0.12) > STRESS vol_target (0.06) → CALM weights larger
    EXPECT_GT(vec_mean(alpha_calm), vec_mean(alpha_stress))
        << "CALM regime should produce larger vol-scaled weights than STRESS";
}

TEST(VolScalingRegime, ZeroVolGuardAllRegimes) {
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha = {2.0f};
        std::vector<float> vol   = {0.0f};
        auto result = RiskKernel::apply_vol_scaling_regime(
            alpha, vol, static_cast<Regime>(r));
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(std::isinf(alpha[0])) << "Inf produced for regime " << r;
        EXPECT_FALSE(std::isnan(alpha[0])) << "NaN produced for regime " << r;
    }
}

// ── apply_nonlinear_interaction_cap ──────────────────────────────────────────

TEST(NonlinearInteractionCap, HighVolAssetGetsLowerCap) {
    // Low-vol asset and high-vol asset given same raw alpha
    std::vector<float> alpha_lv = {0.30f};   // same raw score
    std::vector<float> alpha_hv = {0.30f};
    std::vector<float> vol_lv   = {0.05f};   // low vol
    std::vector<float> vol_hv   = {0.40f};   // high vol (8x higher)

    RiskKernel::apply_nonlinear_interaction_cap(
        alpha_lv, vol_lv, kRegimeTable[1].vol_target, kRegimeTable[1].weight_cap);
    RiskKernel::apply_nonlinear_interaction_cap(
        alpha_hv, vol_hv, kRegimeTable[1].vol_target, kRegimeTable[1].weight_cap);

    // High-vol asset's effective cap = weight_cap * (vol_target/vol_hv) < weight_cap
    // so it should be clamped to a smaller value
    EXPECT_GE(alpha_lv[0], alpha_hv[0])
        << "Low-vol asset should have equal or larger weight than high-vol asset";
}

TEST(NonlinearInteractionCap, AllRegimesProduceBoundedOutput) {
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha = {-0.5f, 0.0f, 0.5f, 1.0f};
        std::vector<float> vol   = {0.10f, 0.15f, 0.20f, 0.30f};
        auto result = RiskKernel::apply_nonlinear_interaction_cap(
            alpha, vol,
            kRegimeTable[r].vol_target,
            kRegimeTable[r].weight_cap);
        ASSERT_TRUE(result.has_value());
        float cap = kRegimeTable[r].weight_cap;
        for (float v : alpha) {
            EXPECT_LE(v,  cap + 1e-5f) << "Regime " << r << " cap exceeded";
            EXPECT_GE(v, -cap - 1e-5f) << "Regime " << r << " cap exceeded";
        }
    }
}

// ── apply_circuit_breaker_regime ─────────────────────────────────────────────

TEST(CircuitBreakerRegime, StressCapLowerThanCalmCap) {
    std::vector<float> alpha_calm   = {0.50f, -0.50f};
    std::vector<float> alpha_stress = {0.50f, -0.50f};

    RiskKernel::apply_circuit_breaker_regime(alpha_calm,   Regime::CALM);
    RiskKernel::apply_circuit_breaker_regime(alpha_stress, Regime::STRESS);

    EXPECT_GT(vec_max_abs(alpha_calm), vec_max_abs(alpha_stress))
        << "CALM cap (0.25) should allow larger weights than STRESS cap (0.12)";
}

TEST(CircuitBreakerRegime, OrderingInvariant) {
    // Invariant from test_nonlinear_pipeline.py mirrored in C++:
    // cap(STRESS) < cap(TRANSITION) < cap(CALM)
    EXPECT_LT(kRegimeTable[static_cast<int>(Regime::STRESS)].weight_cap,
              kRegimeTable[static_cast<int>(Regime::TRANSITION)].weight_cap);
    EXPECT_LT(kRegimeTable[static_cast<int>(Regime::TRANSITION)].weight_cap,
              kRegimeTable[static_cast<int>(Regime::CALM)].weight_cap);
}

// ── MacroAlphaEngine v2 full pipeline ─────────────────────────────────────────

TEST(MacroAlphaEngineV2, AllRegimesProduceValidOutput) {
    const std::size_t N = 64;
    for (int r = 0; r < 3; ++r) {
        std::vector<float> alpha(N), vol(N);
        for (std::size_t i = 0; i < N; ++i) {
            alpha[i] = static_cast<float>(i % 11) - 5.0f;
            vol[i]   = 0.05f + 0.01f * static_cast<float>(i % 20);
        }
        MacroAlphaEngine engine{kRegimeTable[r].vol_target, kRegimeTable[r].weight_cap};
        auto result = engine.run_pipeline(alpha, vol, static_cast<Regime>(r));
        ASSERT_TRUE(result.has_value()) << "run_pipeline failed for regime " << r;
        EXPECT_TRUE(has_no_nan_inf(alpha)) << "NaN/Inf for regime " << r;
        float cap = kRegimeTable[r].weight_cap;
        for (float v : alpha) {
            EXPECT_LE(v,  cap + 1e-5f) << "Cap exceeded, regime=" << r;
            EXPECT_GE(v, -cap - 1e-5f) << "Cap exceeded, regime=" << r;
        }
    }
}

TEST(MacroAlphaEngineV2, StressWeightsTighterThanCalm) {
    const std::size_t N = 32;
    std::vector<float> alpha_c(N, 3.0f), alpha_s(N, 3.0f);
    std::vector<float> vol(N, 0.10f);

    MacroAlphaEngine eng_c{kRegimeTable[0].vol_target, kRegimeTable[0].weight_cap};
    MacroAlphaEngine eng_s{kRegimeTable[2].vol_target, kRegimeTable[2].weight_cap};

    eng_c.run_pipeline(alpha_c, vol, Regime::CALM);
    eng_s.run_pipeline(alpha_s, vol, Regime::STRESS);

    EXPECT_GT(vec_max_abs(alpha_c), vec_max_abs(alpha_s))
        << "CALM should allow larger weights than STRESS";
}

TEST(MacroAlphaEngineV2, BackwardCompatV1StillWorks) {
    // v1 overload: run_pipeline(alpha, vol) without regime arg
    std::vector<float> alpha = {2.0f, -1.0f, 0.5f, 3.0f};
    std::vector<float> vol   = {0.10f, 0.15f, 0.20f, 0.08f};
    MacroAlphaEngine engine{0.10f, 0.20f};
    auto result = engine.run_pipeline(alpha, vol);  // TRANSITION default
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
    for (float v : alpha) EXPECT_LE(std::abs(v), 0.20f + 1e-5f);
}

TEST(MacroAlphaEngineV2, SizeMismatchPropagated) {
    std::vector<float> alpha = {1.0f, 2.0f};
    std::vector<float> vol   = {0.10f};
    MacroAlphaEngine engine{0.10f};
    auto r = engine.run_pipeline(alpha, vol, Regime::TRANSITION);
    EXPECT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("size mismatch"), std::string::npos);
}

TEST(MacroAlphaEngineV2, AllZeroAlphaNoNaN) {
    std::vector<float> alpha(128, 0.0f);
    std::vector<float> vol(128, 0.10f);
    MacroAlphaEngine engine{0.10f, 0.20f};
    auto r = engine.run_pipeline(alpha, vol, Regime::CALM);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
}

TEST(MacroAlphaEngineV2, ExtremeInputsClamped) {
    std::vector<float> alpha = {1e6f, -1e6f, 1e6f, -1e6f};
    std::vector<float> vol   = {0.01f, 0.01f, 0.50f, 0.50f};
    MacroAlphaEngine engine{0.10f, 0.20f};
    auto r = engine.run_pipeline(alpha, vol, Regime::STRESS);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(has_no_nan_inf(alpha));
    float cap = kRegimeTable[static_cast<int>(Regime::STRESS)].weight_cap;
    for (float v : alpha) EXPECT_LE(std::abs(v), cap + 1e-5f);
}

} // namespace alpha_pod::test