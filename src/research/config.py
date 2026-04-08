# src/research/config.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# PodConfig v2 — adds nonlinear causality and regime parameters.

from __future__ import annotations
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class PodConfig:
    """Frozen configuration for the nonlinear macro alpha pipeline v2.

    Attributes:
        VOL_TARGET:          Baseline annualised target vol (overridden by regime).
        WEIGHT_CAP:          Baseline per-asset weight cap (overridden by regime).
        LOOKBACK:            Rolling window for base features (days).
        ZSCORE_CLIP:         Cross-sectional z-score clip (base; tightened in stress).
        MAX_MODEL_PATH:      Path to compiled MAX Engine model.
        ANNUALISE:           Annualisation factor (√252).
        CAUSAL_LOOKBACK:     Rolling window for Transfer Entropy / CCM (bars).
        CAUSAL_N_SURROGATES: AAFT surrogates for linear/nonlinear decomposition.
        CAUSAL_LAG:          TE lag (bars).
        CAUSAL_K_NEIGHBOURS: k-NN parameter for entropy estimation.
        REGIME_N_STATES:     HMM hidden states (3 = calm/transition/stress).
    """

    VOL_TARGET:           Final[float] = 0.10
    WEIGHT_CAP:           Final[float] = 0.20
    LOOKBACK:             Final[int]   = 21
    ZSCORE_CLIP:          Final[float] = 3.0
    MAX_MODEL_PATH:       Final[str]   = "models/alpha_macro.max"
    ANNUALISE:            Final[float] = 252.0 ** 0.5
    CAUSAL_LOOKBACK:      Final[int]   = 252
    CAUSAL_N_SURROGATES:  Final[int]   = 19
    CAUSAL_LAG:           Final[int]   = 1
    CAUSAL_K_NEIGHBOURS:  Final[int]   = 5
    REGIME_N_STATES:      Final[int]   = 3
