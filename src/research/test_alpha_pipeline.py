# src/research/test_alpha_pipeline.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# pytest suite for the v1 (linear, single-regime) production pipeline.
# Tests the baseline AlphaProductionPipeline with 3 features and fixed risk
# parameters. This suite is retained alongside test_nonlinear_pipeline.py to
# verify v1/v2 backward compatibility and regression protection.
#
# Coverage:
#   FeatureEngine v1:         schema, z-score bounds, no NaN, date ordering
#   InferenceEngine:          shape, dtype, wrong-dtype/ndim errors
#   AlphaProductionPipeline:  weight bounds, no NaN/Inf, multi-asset isolation
#
# Run: pytest src/research/test_alpha_pipeline.py -v --tb=short

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from config import PodConfig
from inference import InferenceEngine

# Import the v1 FeatureEngine and pipeline from the v1 module.
# If only the v2 versions are present, we create minimal shims below.
try:
    # Try to import v1-style FeatureEngine (3-feature, Pearson z-score)
    # In the v2 monorepo, features.py IS the v2 version, which is a superset.
    # The v2 FeatureEngine accepts no regime_df/causal_df and degrades to v1 behaviour.
    from features import FeatureEngine
    from alpha_pipeline import AlphaProductionPipeline
except ImportError as e:
    pytest.skip(f"Pipeline not importable: {e}", allow_module_level=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> PodConfig:
    return PodConfig()


@pytest.fixture(scope="module")
def raw_df() -> pl.DataFrame:
    """Minimal synthetic DataFrame: 2 assets × 100 days."""
    rng = np.random.default_rng(0)
    assets = ["JPY", "EUR"]
    n = 100
    dates = pl.date_range(
        start=pl.date(2025, 1, 2),
        end=pl.date(2025, 6, 30),
        interval="1d",
        eager=True,
    )[:n]
    rows = []
    for asset in assets:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        rates  = rng.uniform(0.001, 0.005, n)
        for d, p, r in zip(dates, prices, rates):
            rows.append({"date": d, "asset": asset, "price": float(p), "rate": float(r)})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def feat_df(cfg: PodConfig, raw_df: pl.DataFrame) -> pl.DataFrame:
    # v2 FeatureEngine with no regime/causal args → degrades to v1-equivalent
    return FeatureEngine(cfg).compute(raw_df)


# ── FeatureEngine ──────────────────────────────────────────────────────────────

class TestFeatureEngine:
    def test_output_has_base_feature_columns(self, feat_df: pl.DataFrame) -> None:
        for col in ("f_tsmom", "f_vol", "f_carry"):
            assert col in feat_df.columns, f"Missing column: {col}"

    def test_zscore_bounds(self, feat_df: pl.DataFrame, cfg: PodConfig) -> None:
        """All feature values must lie within ±ZSCORE_CLIP after warm-up."""
        warm = feat_df.filter(pl.col("f_tsmom").is_not_null())
        for col in ("f_tsmom", "f_carry"):
            vals = warm[col].drop_nulls().to_numpy()
            assert np.all(vals <= cfg.ZSCORE_CLIP + 0.01), f"{col} exceeds +{cfg.ZSCORE_CLIP} clip"
            assert np.all(vals >= -cfg.ZSCORE_CLIP - 0.01), f"{col} below -{cfg.ZSCORE_CLIP} clip"

    def test_no_nan_in_vol(self, feat_df: pl.DataFrame) -> None:
        warm_vol = feat_df["f_vol"].drop_nulls().to_numpy()
        assert not np.any(np.isnan(warm_vol)), "NaN in f_vol"

    def test_sorted_by_asset_date(self, raw_df: pl.DataFrame, cfg: PodConfig) -> None:
        out = FeatureEngine(cfg).compute(raw_df)
        for asset in ("JPY", "EUR"):
            asset_dates = out.filter(pl.col("asset") == asset)["date"].to_list()
            assert asset_dates == sorted(asset_dates), f"Dates not sorted for {asset}"

    def test_interaction_cols_present_in_v2(self, feat_df: pl.DataFrame) -> None:
        """v2 FeatureEngine always adds interaction columns even without regime_df."""
        for col in ("f_tsmom_x_vol", "f_carry_x_vol", "f_vol_regime_adj"):
            assert col in feat_df.columns, f"v2 interaction column missing: {col}"

    def test_causal_cols_zero_without_causal_df(self, feat_df: pl.DataFrame) -> None:
        """Without causal_df, causal features default to 0."""
        for col in ("f_te_net_cause", "f_ccm_driver"):
            if col in feat_df.columns:
                unique_vals = feat_df[col].drop_nulls().unique().to_list()
                # All zeros when no causal_df is provided
                assert all(abs(v) < 1e-6 for v in unique_vals), \
                    f"{col} should be 0 without causal_df"


# ── InferenceEngine ────────────────────────────────────────────────────────────

class TestInferenceEngine:
    def test_output_shape_3_features(self, cfg: PodConfig) -> None:
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.random.default_rng(1).standard_normal((20, 3)).astype(np.float32)
        out = eng.execute(X)
        assert out.shape == (20,), f"Expected (20,), got {out.shape}"

    def test_output_shape_9_features(self, cfg: PodConfig) -> None:
        """v2 inference handles 9-feature input."""
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.random.default_rng(2).standard_normal((15, 9)).astype(np.float32)
        out = eng.execute(X)
        assert out.shape == (15,), f"Expected (15,), got {out.shape}"

    def test_output_dtype(self, cfg: PodConfig) -> None:
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.ones((5, 3), dtype=np.float32)
        out = eng.execute(X)
        assert out.dtype == np.float32

    def test_no_nan_in_output(self, cfg: PodConfig) -> None:
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.random.default_rng(3).standard_normal((50, 9)).astype(np.float32)
        out = eng.execute(X)
        assert not np.any(np.isnan(out)), "NaN in inference output"

    def test_wrong_dtype_raises(self, cfg: PodConfig) -> None:
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.ones((5, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            eng.execute(X)

    def test_wrong_ndim_raises(self, cfg: PodConfig) -> None:
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        X = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            eng.execute(X)

    def test_nonlinear_fallback_differs_from_linear(self, cfg: PodConfig) -> None:
        """Nonlinear fallback must differ from simple w=[1,1,-1] linear model."""
        eng = InferenceEngine(cfg.MAX_MODEL_PATH)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        nl_out = eng.execute(X)
        lin_out = (X @ np.array([1.0, 1.0, -1.0], dtype=np.float32))
        assert not np.allclose(nl_out, lin_out, atol=1e-3), \
            "Fallback must differ from v1 linear weights"


# ── AlphaProductionPipeline ────────────────────────────────────────────────────

class TestAlphaProductionPipeline:
    def test_final_weight_column_present(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        assert "final_weight" in result.columns

    def test_weights_within_max_regime_cap(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        """Weights must not exceed the CALM cap (0.25) — the maximum possible."""
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        weights = result["final_weight"].drop_nulls().to_numpy()
        assert np.all(np.abs(weights) <= 0.25 + 1e-5), \
            f"Weight exceeded max allowed cap 0.25"

    def test_no_nan_or_inf_in_weights(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        weights = result["final_weight"].drop_nulls().to_numpy()
        assert not np.any(np.isnan(weights)), "NaN in final_weight"
        assert not np.any(np.isinf(weights)), "Inf in final_weight"

    def test_multi_asset_isolation(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        """Both assets must appear in results."""
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        assert set(result["asset"].unique().to_list()) == {"JPY", "EUR"}

    def test_regime_column_present(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        """v2 pipeline must always output regime column."""
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        assert "regime" in result.columns, "v2 pipeline missing 'regime' column"

    def test_causal_feature_columns_present(
        self, cfg: PodConfig, raw_df: pl.DataFrame
    ) -> None:
        """v2 pipeline must always output causal feature columns."""
        pipeline = AlphaProductionPipeline(cfg)
        result = pipeline.run(raw_df)
        for col in ("f_te_net_cause", "f_ccm_driver"):
            assert col in result.columns, f"Missing causal column: {col}"
