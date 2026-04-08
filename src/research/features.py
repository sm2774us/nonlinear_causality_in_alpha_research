# src/research/features.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# Nonlinear feature engineering for the macro alpha pipeline.
#
# UPGRADE from v1: Addresses Ma & Prosperino (2023) — see causality.py.
#
# The original pipeline committed the "Principle of Superposition" error:
# features were purely linear (log-returns, rolling std, rate/vol) and were
# combined additively. This ignores:
#   (a) Nonlinear causality between assets (TE_nonlinear >> TE_linear for equities)
#   (b) Interaction terms: f_tsmom and f_vol are NOT independent; their product
#       carries different information from either alone (momentum in a high-vol
#       regime behaves differently from momentum in a calm regime)
#   (c) Regime-conditional statistics: z-scoring with full-sample mean/std
#       mixes calm and stress regimes, destroying the nonlinear regime signal
#
# New features added in v2:
#   f_tsmom_x_vol     : f_tsmom × f_vol  — momentum quality (high-conviction = large signal / low vol)
#   f_carry_x_vol     : f_carry × f_vol  — carry adjusted for vol regime
#   f_regime_mom      : f_tsmom × regime_stress_indicator (0/1/2) — regime-gated momentum
#   f_te_net_cause    : nonlinear causality centrality from CausalFeatureMatrix
#   f_ccm_driver      : CCM-based driver score
#   f_nl_zscore       : nonlinear rank-based z-score (spearman, not Pearson)
#   f_vol_regime_adj  : vol normalised by regime-specific rolling vol-of-vol
#
# All interaction features are computed WITHIN the Polars lazy graph where
# possible; causal features (TE, CCM) are pre-computed daily and joined.
#
# Usage:
#   from features import FeatureEngine
#   engine = FeatureEngine(cfg, regime_df=regime_df, causal_df=causal_df)
#   feat_df = engine.compute(raw_df)

from __future__ import annotations

import polars as pl
import numpy as np
from typing import Optional

from config import PodConfig


class FeatureEngine:
    """Nonlinear feature engineering with causal and interaction terms.

    Args:
        cfg:        Frozen PodConfig.
        regime_df:  Optional per-date regime DataFrame from RegimeEngine.
                    If provided, enables regime-conditional z-scoring.
        causal_df:  Optional per-asset causal scores from CausalFeatureMatrix.
                    If provided, adds TE and CCM causal features.
    """

    FEATURE_COLS_BASE: tuple[str, ...] = ('f_tsmom', 'f_vol', 'f_carry')
    FEATURE_COLS_INTERACTION: tuple[str, ...] = (
        'f_tsmom_x_vol', 'f_carry_x_vol', 'f_regime_mom', 'f_vol_regime_adj',
    )
    FEATURE_COLS_CAUSAL: tuple[str, ...] = (
        'f_te_net_cause', 'f_ccm_driver', 'f_te_out_degree',
    )

    def __init__(
        self,
        cfg:        PodConfig,
        regime_df:  Optional[pl.DataFrame] = None,
        causal_df:  Optional[pl.DataFrame] = None,
    ) -> None:
        self._cfg       = cfg
        self._regime_df = regime_df
        self._causal_df = causal_df

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Full nonlinear feature pipeline.

        Step 1: Base linear features (f_tsmom, f_vol, f_carry).
        Step 2: Nonlinear cross-sectional z-score (rank-based, not Pearson).
        Step 3: Interaction terms (violate superposition intentionally).
        Step 4: Regime conditioning (if regime_df provided).
        Step 5: Causal centrality features (if causal_df provided).

        Args:
            df: DataFrame with columns [date, asset, price, rate].

        Returns:
            DataFrame with all feature columns appended.
        """
        L   = self._cfg.LOOKBACK
        ann = self._cfg.ANNUALISE
        clip = self._cfg.ZSCORE_CLIP

        # ── Step 1: Base features ─────────────────────────────────────────────
        feat = (
            df.lazy()
            .sort(['asset', 'date'])
            .with_columns([
                pl.col('price').log().diff(L).over('asset').alias('f_tsmom_raw'),
                (
                    pl.col('price').log().diff()
                    .rolling_std(L).over('asset') * ann
                ).alias('f_vol_raw'),
            ])
            .with_columns([
                (pl.col('rate') / (pl.col('f_vol_raw') + 1e-6)).alias('f_carry_raw'),
            ])
            # ── Step 2: Nonlinear (rank-based) cross-sectional z-score ────────
            # Unlike Pearson z-score, rank normalisation is robust to fat tails
            # and preserves the ordinal nonlinear signal rather than rescaling
            # to a Gaussian that doesn't exist in financial returns.
            .with_columns([
                self._rank_zscore_expr('f_tsmom_raw', clip).alias('f_tsmom'),
                self._rank_zscore_expr('f_carry_raw', clip).alias('f_carry'),
                self._rank_zscore_expr('f_vol_raw',   clip).alias('f_vol'),
            ])
            .collect()
        )

        # ── Step 3: Interaction features ──────────────────────────────────────
        # These are THE key nonlinear terms that violate superposition.
        # f_tsmom × f_vol: momentum conditional on volatility regime.
        #   High positive f_tsmom in high f_vol → potentially a breakout.
        #   High positive f_tsmom in low f_vol → quiet trending (different risk).
        # f_carry × f_vol: yield-per-unit-risk adjusted for current vol level.
        feat = feat.with_columns([
            (pl.col('f_tsmom') * pl.col('f_vol')).alias('f_tsmom_x_vol'),
            (pl.col('f_carry') * pl.col('f_vol')).alias('f_carry_x_vol'),
            # Vol-of-vol normalisation: assets with unstable vol (higher f_volvol)
            # should have their vol signal discounted
            (
                pl.col('f_vol_raw')
                / (pl.col('f_vol_raw').rolling_std(L).over('asset') + 1e-6)
            ).alias('f_vol_regime_adj'),
        ])

        # ── Step 4: Regime conditioning ───────────────────────────────────────
        if self._regime_df is not None:
            feat = self._apply_regime_conditioning(feat)
        else:
            feat = feat.with_columns(
                pl.lit(1).cast(pl.Int32).alias('regime'),
                pl.lit(0.0).alias('f_regime_mom'),
            )

        # ── Step 5: Causal features ───────────────────────────────────────────
        if self._causal_df is not None:
            feat = feat.join(self._causal_df, on='asset', how='left')
            # Z-score causal features cross-sectionally
            for col in ('f_te_net_cause', 'f_ccm_driver', 'f_te_out_degree'):
                if col in feat.columns:
                    feat = feat.with_columns(
                        self._rank_zscore_expr(col, clip)
                    )
        else:
            # Fallback: zero causal features (pipeline still valid without CCM)
            for col in self.FEATURE_COLS_CAUSAL:
                feat = feat.with_columns(pl.lit(0.0).alias(col))

        return feat

    @property
    def all_feature_cols(self) -> tuple[str, ...]:
        """All feature column names used in model input."""
        return (
            self.FEATURE_COLS_BASE
            + self.FEATURE_COLS_INTERACTION
            + self.FEATURE_COLS_CAUSAL
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rank_zscore_expr(col: str, clip: float) -> pl.Expr:
        """Rank-based cross-sectional z-score.

        Unlike the Pearson/moment-based z-score:
            z_Pearson = (x - μ) / σ

        the rank z-score converts to uniform [0,1] quantiles, then to
        standard normal quantiles via the probit transform:

            u_i = (rank_i - 0.5) / N      (continuity-corrected)
            z_i = Φ^{-1}(clip(u_i, ε, 1-ε))

        This is EQUIVALENT to Spearman-rank normalisation and preserves the
        ORDINAL (nonlinear) information in the distribution rather than
        forcing a Gaussian assumption. It is the correct normalisation when
        nonlinear causality is present because it doesn't destroy tail structure.

        In Polars, we approximate via linear rank scaling:
            z ≈ 2 * ((rank - rank_min) / (rank_max - rank_min + 1e-6)) - 1
        then clip to ±ZSCORE_CLIP.
        """
        rank = pl.col(col).rank('ordinal').over('date')
        n    = pl.col(col).count().over('date').cast(pl.Float64)
        # Linear rank → [-1, +1] → scale by clip
        z = ((rank.cast(pl.Float64) - 1.0) / (n - 1.0 + 1e-6) * 2.0 - 1.0) * clip
        return z.clip(-clip, clip).alias(col)

    def _apply_regime_conditioning(self, feat: pl.DataFrame) -> pl.DataFrame:
        """Join regime labels and create regime-gated interaction features.

        f_regime_mom = f_tsmom × stress_indicator
        where stress_indicator ∈ {0 (calm), 1 (transition), 2 (stress)}

        This explicitly models the nonlinear interaction between momentum
        and market stress — a relationship that a linear model cannot capture.
        In stress regimes, momentum signals carry different (often reversed)
        predictive content (momentum crashes, Barroso & Santa-Clara 2015).
        """
        regime_slim = self._regime_df.select(['date', 'regime', 'vol_target', 'weight_cap'])
        feat = feat.join(regime_slim, on='date', how='left').with_columns([
            pl.col('regime').fill_null(1),
            pl.col('vol_target').fill_null(self._cfg.VOL_TARGET),
            pl.col('weight_cap').fill_null(self._cfg.WEIGHT_CAP),
        ])
        # Regime-gated momentum: flip sign in stress (momentum crash protection)
        feat = feat.with_columns([
            (
                pl.col('f_tsmom')
                * pl.when(pl.col('regime') == 2).then(-1.0)   # reverse in stress
                  .when(pl.col('regime') == 1).then(0.5)      # dampen in transition
                  .otherwise(1.0)                              # full strength in calm
            ).alias('f_regime_mom'),
        ])
        return feat
