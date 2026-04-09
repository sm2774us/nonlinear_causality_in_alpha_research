// src/alpha_engine/bindings.cpp
// Copyright 2026 AlphaPod Contributors. All Rights Reserved.
//
// nanobind Python extension — v2 (regime-aware).
//
// New in v2:
//   - Regime enum exposed to Python: alpha_engine_cpp.Regime.CALM / TRANSITION / STRESS
//   - run_pipeline() accepts an optional regime argument (default: TRANSITION)
//   - Regime parameter maps directly to kRegimeTable in RiskKernel.ixx

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

//import AlphaPod.MacroAlphaEngine;
#include "MacroAlphaEngine.h"

namespace nb = nanobind;
using FloatArr = nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(alpha_engine_cpp, m) {
    m.doc() = "AlphaPod C++26 regime-conditional risk engine (v2).";

    // ── Expose Regime enum ─────────────────────────────────────────────────────
    nb::enum_<alpha_pod::Regime>(m, "Regime",
        "Market regime for regime-conditional risk kernel.")
        .value("CALM",       alpha_pod::Regime::CALM,       "Low vol, capture carry")
        .value("TRANSITION", alpha_pod::Regime::TRANSITION, "Baseline parameters")
        .value("STRESS",     alpha_pod::Regime::STRESS,     "De-risk; tight caps")
        .export_values();

    // ── MacroAlphaEngine ───────────────────────────────────────────────────────
    nb::class_<alpha_pod::MacroAlphaEngine>(m, "MacroAlphaEngine",
        "Regime-conditional SIMD alpha risk pipeline (v2).")
        .def(nb::init<float, float>(),
             nb::arg("vol_target") = alpha_pod::kRegimeTable[1].vol_target,
             nb::arg("weight_cap") = alpha_pod::kRegimeTable[1].weight_cap)

        .def_prop_ro("vol_target", &alpha_pod::MacroAlphaEngine::vol_target)
        .def_prop_ro("weight_cap", &alpha_pod::MacroAlphaEngine::weight_cap)

        // v2: regime-conditional pipeline
        .def("run_pipeline",
            [](alpha_pod::MacroAlphaEngine& self,
               FloatArr alpha,
               FloatArr vol,
               alpha_pod::Regime regime) {
                auto r = self.run_pipeline(
                    std::span<float>(alpha.data(), alpha.size()),
                    std::span<const float>(vol.data(), vol.size()),
                    regime
                );
                if (!r) throw std::runtime_error(r.error());
            },
            nb::arg("alpha"),
            nb::arg("f_vol"),
            nb::arg("regime") = alpha_pod::Regime::TRANSITION,
            "Regime-conditional rank-z-score → vol-scale → nonlinear cap → circuit breaker.\n"
            "alpha is modified in-place. regime defaults to TRANSITION (v1-compatible).")

        // v1 backward-compatible overload (no regime arg)
        .def("run_pipeline_v1",
            [](alpha_pod::MacroAlphaEngine& self, FloatArr alpha, FloatArr vol) {
                auto r = self.run_pipeline(
                    std::span<float>(alpha.data(), alpha.size()),
                    std::span<const float>(vol.data(), vol.size())
                );
                if (!r) throw std::runtime_error(r.error());
            },
            nb::arg("alpha"),
            nb::arg("f_vol"),
            "v1 backward-compatible pipeline (TRANSITION regime, Pearson-equivalent).");
}
