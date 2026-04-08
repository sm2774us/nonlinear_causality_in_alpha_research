# src/research/inference.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# MAX Engine inference wrapper — v2 (nonlinear, 9-feature).
#
# Supports both:
#   - Production: MAX Engine InferenceSession loading a compiled .max model
#     (GBM or transformer trained on the 9-feature nonlinear set)
#   - Dev/test: NonlinearFallbackModel (sigmoid-gated base + interactions)
#
# The v2 model consumes F=9 features (vs 3 in v1):
#   [f_tsmom, f_carry, f_vol,
#    f_tsmom_x_vol, f_carry_x_vol, f_regime_mom, f_vol_regime_adj,
#    f_te_net_cause, f_ccm_driver]
#
# Usage:
#   from inference import InferenceEngine
#   eng = InferenceEngine("models/alpha_macro_v2.max")
#   raw_alpha = eng.execute(X)   # X: float32 (N, 9)

from __future__ import annotations

import warnings

import numpy as np

try:
    import max.engine as _max_engine  # type: ignore[import]
    _MAX_AVAILABLE = True
except ImportError:
    _MAX_AVAILABLE = False


class InferenceEngine:
    """Thin wrapper around a MAX Engine InferenceSession.

    Args:
        model_path: Path to the compiled .max model file.
                    Accepts both v1 (3-feature) and v2 (9-feature) models;
                    the fallback adapts to whichever feature count is passed.

    Raises:
        ValueError: On bad input dtype/shape in execute().
    """

    # v2 default feature count; fallback adapts if fewer features are passed
    N_FEATURES_V2: int = 9
    N_FEATURES_V1: int = 3

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        if _MAX_AVAILABLE:
            session = _max_engine.InferenceSession()
            self._model = session.load(model_path)
        else:
            warnings.warn(
                "MAX Engine not found; using NonlinearFallbackModel. "
                "Do NOT use in production.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, X: np.ndarray) -> np.ndarray:
        """Forward inference → 1-D float32 alpha score vector.

        Args:
            X: Float32 array of shape (N, F) — N assets × F features.
               F may be 3 (v1 features) or 9 (v2 features).

        Returns:
            Float32 ndarray of shape (N,).

        Raises:
            ValueError: If X is not float32 or not 2-D.
        """
        if X.dtype != np.float32:
            raise ValueError(f"Expected float32 input; got {X.dtype}")
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D input; got shape {X.shape}")

        if self._model is not None:
            output: dict[str, np.ndarray] = self._model.execute(input=X)
            return output["output"].astype(np.float32, copy=False).ravel()

        # Nonlinear fallback adapts to feature count
        return self._nonlinear_fallback(X)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _nonlinear_fallback(X: np.ndarray) -> np.ndarray:
        """Nonlinear fallback: sigmoid-gated base + interactions + causal.

        For F=3 (v1 features):  sigmoid(w_b · [tsmom, carry, vol])
        For F=9 (v2 features):  sigmoid(base) + w_cross · cross + w_causal · causal

        This is NOT a trained model — it is a deterministic fallback for
        dev/test that demonstrates the correct nonlinear structure.
        Production MUST use MAX Engine with a walk-forward-validated model.
        """
        N, F = X.shape

        # Base weights: [f_tsmom, f_carry, f_vol]
        w_base = np.array([1.0, 0.8, -0.5], dtype=np.float32)

        if F >= 3:
            base_linear = (X[:, :3] * w_base[:3]).sum(axis=1)
        else:
            base_linear = np.zeros(N, dtype=np.float32)

        # Sigmoid gating — prevents superposition blow-up
        alpha = (2.0 / (1.0 + np.exp(-np.clip(base_linear, -10.0, 10.0))) - 1.0).astype(np.float32)

        if F >= 7:
            # Cross/interaction features: f_tsmom_x_vol, f_carry_x_vol, f_regime_mom
            w_cross = np.array([0.4, 0.3, 0.5], dtype=np.float32)
            alpha += (X[:, 3:6] * w_cross).sum(axis=1).astype(np.float32)

        if F >= 9:
            # Causal features: f_te_net_cause, f_ccm_driver
            w_causal = np.array([0.3, 0.2], dtype=np.float32)
            alpha += (X[:, 7:9] * w_causal).sum(axis=1).astype(np.float32)

        return alpha
