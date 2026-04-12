// src/alpha_engine/bindings.cpp
// Copyright 2026 AlphaPod Contributors. All Rights Reserved.
//
// nanobind Python extension — v2 (regime-aware).
//
// New in v2:
//   - Regime enum exposed to Python: alpha_engine_cpp.Regime.CALM / TRANSITION / STRESS
//   - run_pipeline() accepts an optional regime argument (default: TRANSITION)
//   - Regime parameter maps directly to kRegimeTable in RiskKernel.ixx
// Plain TU — NOT a C++ named module interface.
// A Python nanobind extension must never declare 'export module X;' because
// CMake requires every such file to be in FILE_SET CXX_MODULES (like .ixx
// files). Declaring it here while registered as a plain add_library source
// causes cmake_ninja_dyndep to fail:
//   provides AlphaPod.Bindings but not found in FILE_SET CXX_MODULES
// A plain TU CAN use C++26 'import': clang-scan-deps detects them and
// CMake passes -fmodule-file=... at compile time via dyndep.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <span>

// C++26 named-module imports (resolved at compile time via dyndep)
import AlphaPod.MacroAlphaEngine;
import AlphaPod.RiskKernel;

namespace nb = nanobind;

// Use explicit aliases to force the compiler to resolve the module symbols early
using MacroAlphaEngine = alpha_pod::MacroAlphaEngine;
using Regime = alpha_pod::Regime;

using FloatArr = nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(alpha_engine_cpp, m) {
    m.doc() = "AlphaPod C++26 regime-conditional risk engine (v2).";

    // ── Expose Regime enum ─────────────────────────────────────────────────────
    nb::enum_<Regime>(m, "Regime",
        "Market regime for regime-conditional risk kernel.")
        .value("CALM",       Regime::CALM)
        .value("TRANSITION", Regime::TRANSITION)
        .value("STRESS",     Regime::STRESS)
        .export_values();

    // ── MacroAlphaEngine ───────────────────────────────────────────────────────
    nb::class_<MacroAlphaEngine>(m, "MacroAlphaEngine")
        // FIX: Use hardcoded literals for defaults in the binding layer.
        // Referring to kRegimeTable[1] here can cause "module visibility" errors in Clang 19.
        .def(nb::init<float, float>(),
             nb::arg("vol_target") = 0.10f,
             nb::arg("weight_cap") = 0.20f)

        .def_prop_ro("vol_target", &MacroAlphaEngine::vol_target)
        .def_prop_ro("weight_cap", &MacroAlphaEngine::weight_cap)

        // v2: regime-conditional pipeline
        .def("run_pipeline",
            [](MacroAlphaEngine& self,
               FloatArr alpha,
               FloatArr vol,
               Regime regime) {
                auto r = self.run_pipeline(
                    std::span<float>(alpha.data(), alpha.size()),
                    std::span<const float>(vol.data(), vol.size()),
                    regime
                );
                if (!r) throw std::runtime_error(r.error());
            },
            nb::arg("alpha"),
            nb::arg("f_vol"),
            nb::arg("regime") = Regime::TRANSITION,
            "Regime-conditional: rank-z-score -> vol-scale -> nonlinear cap -> circuit breaker. "
            "alpha modified in-place. regime defaults to TRANSITION (v1-compatible).")

        // v1 backward-compatible overload (no regime arg)
        .def("run_pipeline_v1",
            [](MacroAlphaEngine& self, FloatArr alpha, FloatArr vol) {
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