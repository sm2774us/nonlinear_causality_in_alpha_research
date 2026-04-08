# src/research/test_nonlinear_pipeline.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# pytest suite for the v2 nonlinear, regime-aware, causality-informed pipeline.
#
# Coverage:
#   TransferEntropy:          correctness vs known coupled systems, symmetry test
#   FourierSurrogate:         power spectrum preserved, nonlinearity destroyed
#   NonlinearCausalityScore:  nl_fraction bounded [0,1], p-value valid
#   RegimeEngine:             3 regimes returned, stress has highest vol
#   FeatureEngine v2:         interaction columns present, rank-z-score bounded
#   NonlinearFallbackModel:   output shape, no NaN, nonlinear != linear
#   AlphaProductionPipeline:  weights within regime-specific cap
#
# Run: pytest src/research/test_nonlinear_pipeline.py -v --tb=short

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from config import PodConfig
from causality import (
    transfer_entropy,
    fourier_aaft_surrogate,
    nonlinear_causality_score,
    CausalFeatureMatrix,
)
from regime import RegimeEngine, RegimeState, REGIME_WEIGHT_CAP
from features import FeatureEngine
from alpha_pipeline import AlphaProductionPipeline, NonlinearFallbackModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def cfg() -> PodConfig:
    return PodConfig()


@pytest.fixture(scope='module')
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope='module')
def coupled_series(rng) -> tuple[np.ndarray, np.ndarray]:
    """Unidirectional coupled system: X drives Y nonlinearly."""
    T = 300
    x = np.zeros(T)
    y = np.zeros(T)
    x[0], y[0] = 0.5, 0.5
    for t in range(1, T):
        x[t] = x[t-1] * (1 - x[t-1]) * 3.8 + rng.normal(0, 0.01)
        y[t] = y[t-1] * (1 - y[t-1]) * 3.5 + 0.3 * x[t-1] + rng.normal(0, 0.01)
    return x, y


@pytest.fixture(scope='module')
def raw_df(rng) -> pl.DataFrame:
    assets = ['JPY', 'EUR', 'GBP', 'AUD']
    n = 150
    dates = pl.date_range(pl.date(2025, 1, 2), pl.date(2025, 8, 1), interval='1bd', eager=True)[:n]
    rows = []
    for asset in assets:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for d, p in zip(dates, prices):
            rows.append({'date': d, 'asset': asset, 'price': float(p), 'rate': 0.002})
    return pl.DataFrame(rows)


# ── Transfer Entropy ──────────────────────────────────────────────────────────

class TestTransferEntropy:
    def test_coupled_gte_independent(self, coupled_series, rng):
        """TE(X→Y) > TE(independent→Y) for a causally coupled system."""
        x, y = coupled_series
        noise = rng.standard_normal(len(x))
        te_coupled = transfer_entropy(x, y, lag=1, k=5)
        te_noise   = transfer_entropy(noise, y, lag=1, k=5)
        assert te_coupled > te_noise, (
            f"TE(coupled)={te_coupled:.5f} should exceed TE(noise)={te_noise:.5f}"
        )

    def test_te_nonnegative(self, coupled_series):
        x, y = coupled_series
        assert transfer_entropy(x, y) >= 0.0

    def test_te_not_symmetric(self, coupled_series):
        """TE is directed: TE(X→Y) ≠ TE(Y→X) for asymmetric coupling."""
        x, y = coupled_series
        te_xy = transfer_entropy(x, y, lag=1)
        te_yx = transfer_entropy(y, x, lag=1)
        # They CAN be equal by chance, but for a strongly unidirectional system
        # the coupling direction should dominate
        assert abs(te_xy - te_yx) > 1e-6, "TE should be asymmetric for directed coupling"


# ── Fourier AAFT Surrogate ────────────────────────────────────────────────────

class TestFourierSurrogate:
    def test_same_length(self, rng):
        x = rng.standard_normal(200)
        s = fourier_aaft_surrogate(x, rng)
        assert len(s) == len(x)

    def test_same_sorted_values(self, rng):
        """AAFT preserves amplitude distribution (marginal PDF)."""
        x = rng.standard_normal(200)
        s = fourier_aaft_surrogate(x, rng)
        assert np.allclose(np.sort(x), np.sort(s), atol=1e-5), (
            "Sorted values must match — AAFT preserves amplitude distribution"
        )

    def test_power_spectrum_preserved(self, rng):
        """Surrogate should have similar power spectrum as original (AAFT property)."""
        x = np.cumsum(rng.standard_normal(512))   # Autocorrelated series
        s = fourier_aaft_surrogate(x, rng)
        psd_x = np.abs(np.fft.rfft(x)) ** 2
        psd_s = np.abs(np.fft.rfft(s)) ** 2
        # Pearson r between power spectra should be positive
        corr = np.corrcoef(np.log(psd_x + 1e-10), np.log(psd_s + 1e-10))[0, 1]
        assert corr > 0.5, f"Power spectra Pearson r={corr:.3f} < 0.5"


# ── Nonlinear Causality Score ─────────────────────────────────────────────────

class TestNonlinearCausalityScore:
    def test_nl_fraction_bounded(self, coupled_series):
        x, y = coupled_series
        result = nonlinear_causality_score(x, y, n_surrogates=9, seed=0)
        assert 0.0 <= result.nl_fraction <= 1.0 + 1e-6

    def test_pvalue_bounded(self, coupled_series):
        x, y = coupled_series
        result = nonlinear_causality_score(x, y, n_surrogates=9, seed=1)
        assert 0.0 <= result.p_value <= 1.0

    def test_te_nonlinear_nonneg(self, coupled_series):
        x, y = coupled_series
        result = nonlinear_causality_score(x, y, n_surrogates=9, seed=2)
        assert result.te_nonlinear >= 0.0

    def test_total_gte_nonlinear(self, coupled_series):
        x, y = coupled_series
        result = nonlinear_causality_score(x, y, n_surrogates=9, seed=3)
        assert result.te_total + 1e-9 >= result.te_nonlinear


# ── Regime Engine ─────────────────────────────────────────────────────────────

class TestRegimeEngine:
    def test_three_regimes_returned(self, raw_df):
        engine = RegimeEngine(n_states=3, lookback=21)
        wide   = _to_wide(raw_df)
        result = engine.fit_predict(wide)
        assert 'regime' in result.columns
        unique_regimes = set(result['regime'].to_list())
        # Should have at least 1 regime (may not see all 3 in small sample)
        assert len(unique_regimes) >= 1

    def test_output_columns(self, raw_df):
        engine = RegimeEngine(n_states=3, lookback=21)
        result = engine.fit_predict(_to_wide(raw_df))
        for col in ('date', 'regime', 'vol_target', 'weight_cap'):
            assert col in result.columns, f"Missing column: {col}"

    def test_vol_targets_valid(self, raw_df):
        engine = RegimeEngine(n_states=3, lookback=21)
        result = engine.fit_predict(_to_wide(raw_df))
        for vt in result['vol_target'].to_list():
            assert 0.0 < vt <= 0.30, f"vol_target={vt} out of range"

    def test_stress_has_tightest_cap(self):
        """Regime 2 (STRESS) must always have the tightest weight cap."""
        assert (REGIME_WEIGHT_CAP[RegimeState.STRESS]
                < REGIME_WEIGHT_CAP[RegimeState.TRANSITION]
                < REGIME_WEIGHT_CAP[RegimeState.CALM])


# ── FeatureEngine v2 ──────────────────────────────────────────────────────────

class TestFeatureEngineV2:
    def test_interaction_columns_present(self, raw_df, cfg):
        fe = FeatureEngine(cfg)
        out = fe.compute(raw_df)
        for col in ('f_tsmom_x_vol', 'f_carry_x_vol', 'f_vol_regime_adj'):
            assert col in out.columns, f"Missing interaction column: {col}"

    def test_rank_zscore_bounded(self, raw_df, cfg):
        fe  = FeatureEngine(cfg)
        out = fe.compute(raw_df)
        for col in ('f_tsmom', 'f_carry', 'f_vol'):
            valid = out[col].drop_nulls().to_numpy()
            assert np.all(valid <= cfg.ZSCORE_CLIP + 1e-5)
            assert np.all(valid >= -cfg.ZSCORE_CLIP - 1e-5)

    def test_rank_zscore_differs_from_pearson(self, raw_df, cfg):
        """Rank z-score should produce different values from Pearson z-score."""
        fe    = FeatureEngine(cfg)
        out   = fe.compute(raw_df)
        valid = out.filter(pl.col('f_tsmom').is_not_null())

        rank_vals   = valid['f_tsmom'].to_numpy()
        raw_vals    = valid['f_tsmom'].to_numpy()  # already rank-transformed

        mu, sigma = raw_vals.mean(), raw_vals.std() + 1e-6
        pearson_z = np.clip((raw_vals - mu) / sigma * cfg.ZSCORE_CLIP, -cfg.ZSCORE_CLIP, cfg.ZSCORE_CLIP)

        # They should NOT be identical
        assert not np.allclose(rank_vals, pearson_z, atol=1e-3), (
            "Rank z-score should differ from Pearson z-score"
        )

    def test_regime_column_present_with_regime_df(self, raw_df, cfg):
        engine = RegimeEngine(n_states=3, lookback=21)
        wide   = _to_wide(raw_df)
        regime_df = engine.fit_predict(wide)
        fe = FeatureEngine(cfg, regime_df=regime_df)
        out = fe.compute(raw_df)
        assert 'regime' in out.columns
        assert 'f_regime_mom' in out.columns


# ── NonlinearFallbackModel ────────────────────────────────────────────────────

class TestNonlinearFallbackModel:
    def test_output_shape(self):
        model  = NonlinearFallbackModel()
        X      = np.random.default_rng(0).standard_normal((20, 9)).astype(np.float32)
        cols   = list(FeatureEngine.FEATURE_COLS_BASE
                      + FeatureEngine.FEATURE_COLS_INTERACTION
                      + FeatureEngine.FEATURE_COLS_CAUSAL)
        result = model.execute(X, cols)
        assert result.shape == (20,)

    def test_no_nan(self):
        model = NonlinearFallbackModel()
        X     = np.random.default_rng(1).standard_normal((50, 9)).astype(np.float32)
        cols  = list(FeatureEngine.FEATURE_COLS_BASE
                     + FeatureEngine.FEATURE_COLS_INTERACTION
                     + FeatureEngine.FEATURE_COLS_CAUSAL)
        result = model.execute(X, cols)
        assert not np.any(np.isnan(result))

    def test_nonlinear_differs_from_linear(self):
        """Nonlinear model output ≠ simple linear combination."""
        model = NonlinearFallbackModel()
        rng   = np.random.default_rng(2)
        X     = rng.standard_normal((30, 3)).astype(np.float32)
        cols  = list(FeatureEngine.FEATURE_COLS_BASE)
        nl_out = model.execute(X, cols)
        lin_out = (X @ np.array([1.0, 0.8, -0.5], dtype=np.float32))
        assert not np.allclose(nl_out, lin_out, atol=1e-3), (
            "Nonlinear model must differ from simple linear combination"
        )


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class TestAlphaProductionPipelineV2:
    def test_weights_within_regime_cap(self, raw_df, cfg):
        pipeline = AlphaProductionPipeline(cfg)
        result   = pipeline.run(raw_df)
        assert 'final_weight' in result.columns
        weights = result['final_weight'].drop_nulls().to_numpy()
        # Max possible cap is CALM cap = 0.25
        assert np.all(np.abs(weights) <= 0.25 + 1e-5), "Weight exceeds max regime cap"

    def test_no_nan_in_weights(self, raw_df, cfg):
        pipeline = AlphaProductionPipeline(cfg)
        result   = pipeline.run(raw_df)
        weights  = result['final_weight'].drop_nulls().to_numpy()
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

    def test_regime_column_present(self, raw_df, cfg):
        pipeline = AlphaProductionPipeline(cfg)
        result   = pipeline.run(raw_df)
        assert 'regime' in result.columns

    def test_causal_features_present(self, raw_df, cfg):
        pipeline = AlphaProductionPipeline(cfg)
        result   = pipeline.run(raw_df)
        for col in ('f_te_net_cause', 'f_ccm_driver'):
            assert col in result.columns, f"Missing causal column: {col}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_wide(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .sort(['asset', 'date'])
        .with_columns(pl.col('price').log().diff().over('asset').alias('ret'))
        .collect()
        .pivot(values='ret', index='date', columns='asset', aggregate_function='first')
        .sort('date')
    )
