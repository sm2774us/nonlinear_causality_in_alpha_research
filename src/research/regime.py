# src/research/regime.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# Regime detection for nonlinear causality conditioning.
#
# Ma & Prosperino (2023) show nonlinear causality is regime-dependent:
# the nonlinear component of TE is significantly elevated during stress
# regimes (volatility clustering, tail co-movements) versus calm regimes.
# A production pipeline must therefore condition ALL features and risk
# parameters on the current regime — not apply a single fixed model.
#
# This module provides:
#   RegimeEngine  — Gaussian HMM over realised vol and return skewness.
#   RegimeState   — Enum: CALM (1), TRANSITION (2), STRESS (3).
#   RegimeFeatures — Per-asset, per-date regime context features.
#
# The regime label feeds:
#   1. features.py  — regime-specific z-score windows and feature interactions
#   2. MacroAlphaEngine.ixx — regime-conditional vol-target and weight cap
#
# Usage:
#   from regime import RegimeEngine
#   engine = RegimeEngine(n_states=3)
#   regime_df = engine.fit_predict(returns_df)

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import polars as pl


class RegimeState(IntEnum):
    """Market regime labels. Higher = more stressed."""
    CALM       = 0
    TRANSITION = 1
    STRESS     = 2


# Per-regime risk parameters broadcast to C++ MacroAlphaEngine
REGIME_VOL_TARGET: Final[dict[int, float]] = {
    RegimeState.CALM:       0.12,   # Slightly more aggressive in calm
    RegimeState.TRANSITION: 0.10,   # Baseline
    RegimeState.STRESS:     0.06,   # De-risk hard in stress
}

REGIME_WEIGHT_CAP: Final[dict[int, float]] = {
    RegimeState.CALM:       0.25,
    RegimeState.TRANSITION: 0.20,
    RegimeState.STRESS:     0.12,   # Hard cap tightens in stress
}


@dataclass
class RegimeEngine:
    """Gaussian Hidden Markov Model for market regime detection.

    Observations are a 4-D vector per date:
        [realised_vol, cross_sectional_dispersion, average_skewness, avg_kurtosis]

    These four dimensions capture:
        - Level of volatility (elevation)
        - Breadth of stress (dispersion across assets)
        - Tail asymmetry (skewness)
        - Fat tails (kurtosis — nonlinear co-movement signature)

    The HMM uses Baum-Welch EM for parameter estimation. At inference time,
    Viterbi decoding gives the MAP state sequence.

    Attributes:
        n_states:     Number of hidden states (default: 3 = calm/transition/stress).
        n_iter:       EM iterations for fitting.
        lookback:     Rolling window for computing obs features.
        tol:          EM convergence tolerance.
    """

    n_states:  int   = 3
    n_iter:    int   = 100
    lookback:  int   = 21
    tol:       float = 1e-4

    def __post_init__(self) -> None:
        # HMM parameters (initialised in fit)
        self._A: np.ndarray | None = None    # Transition matrix (n_states × n_states)
        self._mu: np.ndarray | None = None   # Emission means (n_states × n_obs)
        self._sigma: np.ndarray | None = None  # Emission covariances (n_states × n_obs × n_obs)
        self._pi: np.ndarray | None = None   # Initial state distribution
        self._state_order: list[int] | None = None  # Maps HMM state → RegimeState

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_predict(self, returns_df: pl.DataFrame) -> pl.DataFrame:
        """Fit HMM and return per-date regime labels.

        Args:
            returns_df: Wide DataFrame of daily log-returns (rows=dates, cols=assets).

        Returns:
            DataFrame with columns [date, regime, vol_target, weight_cap,
            obs_vol, obs_disp, obs_skew, obs_kurt].
        """
        obs, dates = self._build_observations(returns_df)
        self._fit(obs)
        states = self._viterbi(obs)
        regimes = self._order_states(obs, states)
        return self._build_output(dates, regimes, obs)

    def predict(self, returns_df: pl.DataFrame) -> pl.DataFrame:
        """Predict regimes on new data using already-fitted HMM parameters.

        Raises:
            RuntimeError: If fit_predict has not been called yet.
        """
        if self._A is None:
            raise RuntimeError("RegimeEngine must be fitted first via fit_predict().")
        obs, dates = self._build_observations(returns_df)
        states = self._viterbi(obs)
        regimes = self._order_states(obs, states)
        return self._build_output(dates, regimes, obs)

    # ── Observation Builder ───────────────────────────────────────────────────

    def _build_observations(
        self, returns_df: pl.DataFrame
    ) -> tuple[np.ndarray, list]:
        """Convert wide returns DataFrame to HMM observation matrix.

        Returns:
            Tuple of (obs array (T, 4), dates list).
        """
        asset_cols = [c for c in returns_df.columns if c != 'date']
        dates = returns_df['date'].to_list() if 'date' in returns_df.columns else list(range(len(returns_df)))
        R = returns_df.select(asset_cols).to_numpy().astype(np.float64)
        T, N = R.shape
        L = self.lookback

        obs = np.full((T, 4), np.nan)
        for t in range(L, T):
            window = R[t - L:t, :]
            valid  = ~np.isnan(window).all(axis=0)
            if valid.sum() < 2:
                continue
            w = window[:, valid]
            daily_vols  = w.std(axis=1)
            avg_ret     = w.mean(axis=1)

            obs[t, 0] = daily_vols.mean() * np.sqrt(252)          # Ann. vol level
            obs[t, 1] = w.std(axis=0).std()                        # Cross-asset vol dispersion
            # Per-asset rolling skewness and kurtosis
            skews = np.array([
                self._skewness(w[:, j]) for j in range(w.shape[1])
            ])
            kurts = np.array([
                self._excess_kurtosis(w[:, j]) for j in range(w.shape[1])
            ])
            obs[t, 2] = float(np.nanmean(skews)) if len(skews) > 0 else 0.0
            obs[t, 3] = float(np.nanmean(kurts)) if len(kurts) > 0 else 0.0

        # Drop NaN rows
        valid_rows = ~np.isnan(obs).any(axis=1)
        obs_clean  = obs[valid_rows]
        dates_clean = [d for d, v in zip(dates, valid_rows) if v]

        # Standardise observations for HMM stability
        self._obs_mean = obs_clean.mean(axis=0)
        self._obs_std  = obs_clean.std(axis=0) + 1e-8
        obs_scaled = (obs_clean - self._obs_mean) / self._obs_std

        return obs_scaled, dates_clean

    # ── Gaussian HMM EM ───────────────────────────────────────────────────────

    def _fit(self, obs: np.ndarray) -> None:
        """Baum-Welch EM for Gaussian HMM parameter estimation."""
        T, D = obs.shape
        K = self.n_states

        # Initialise with k-means
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(obs, K, minit='points', seed=42)

        self._pi = np.bincount(labels, minlength=K).astype(float) + 1.0
        self._pi /= self._pi.sum()

        self._A = np.full((K, K), 1.0 / K)

        self._mu    = np.array([obs[labels == k].mean(axis=0) if (labels == k).any()
                                else obs.mean(axis=0) for k in range(K)])
        self._sigma = np.array([np.cov(obs[labels == k].T) if (labels == k).sum() > D
                                else np.eye(D) * 0.1 for k in range(K)])

        log_lik_prev = -np.inf
        for _ in range(self.n_iter):
            # E-step
            log_B = self._log_emission(obs)   # (T, K)
            alpha, log_lik = self._forward(log_B)
            beta  = self._backward(log_B)
            gamma = self._compute_gamma(alpha, beta)
            xi    = self._compute_xi(alpha, beta, log_B)

            # M-step
            self._pi = gamma[0] + 1e-10
            self._pi /= self._pi.sum()

            A_new = xi.sum(axis=0) + 1e-10
            self._A = A_new / A_new.sum(axis=1, keepdims=True)

            for k in range(K):
                Nk = gamma[:, k].sum() + 1e-10
                self._mu[k] = (gamma[:, k:k+1] * obs).sum(axis=0) / Nk
                diff = obs - self._mu[k]
                self._sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk
                # Regularise
                self._sigma[k] += np.eye(D) * 1e-4

            if abs(log_lik - log_lik_prev) < self.tol:
                break
            log_lik_prev = log_lik

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log-likelihood of observations under each Gaussian state."""
        from scipy.stats import multivariate_normal
        T, D = obs.shape
        K = self.n_states
        log_B = np.zeros((T, self.n_states))
        for k in range(K):
            try:
                log_B[:, k] = multivariate_normal.logpdf(
                    obs, mean=self._mu[k], cov=self._sigma[k], allow_singular=True
                )
            except Exception:
                log_B[:, k] = -1e10
        return log_B

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """Log-scaled forward algorithm."""
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self._pi + 1e-300) + log_B[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t-1] + np.log(self._A[:, k] + 1e-300))
                    + log_B[t, k]
                )
        log_lik = np.logaddexp.reduce(log_alpha[-1])
        return log_alpha, float(log_lik)

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """Log-scaled backward algorithm."""
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    np.log(self._A[k, :] + 1e-300) + log_B[t+1] + log_beta[t+1]
                )
        return log_beta

    def _compute_gamma(
        self, log_alpha: np.ndarray, log_beta: np.ndarray
    ) -> np.ndarray:
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def _compute_xi(
        self,
        log_alpha: np.ndarray,
        log_beta:  np.ndarray,
        log_B:     np.ndarray,
    ) -> np.ndarray:
        T, K = log_B.shape
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + np.log(self._A[i, j] + 1e-300)
                        + log_B[t+1, j]
                        + log_beta[t+1, j]
                    )
            # Normalise
            log_norm = np.logaddexp.reduce(xi[t].ravel())
            xi[t] = np.exp(xi[t] - log_norm)
        return xi

    def _viterbi(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi MAP state sequence decoding."""
        T, K = obs.shape[0], self.n_states
        log_B = self._log_emission(obs)
        delta = np.full((T, K), -np.inf)
        psi   = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self._pi + 1e-300) + log_B[0]
        for t in range(1, T):
            for k in range(K):
                scores       = delta[t-1] + np.log(self._A[:, k] + 1e-300)
                psi[t, k]    = np.argmax(scores)
                delta[t, k]  = scores[psi[t, k]] + log_B[t, k]
        # Traceback
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def _order_states(self, obs: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Re-maps HMM state indices to RegimeState by ascending vol level."""
        K = self.n_states
        vol_by_state = [obs[states == k, 0].mean() if (states == k).any() else 0.0
                        for k in range(K)]
        order = np.argsort(vol_by_state)   # ascending vol → CALM, TRANSITION, STRESS
        remap = np.empty(K, dtype=int)
        for rank, hmm_k in enumerate(order):
            remap[hmm_k] = rank
        self._state_order = remap.tolist()
        return remap[states]

    def _build_output(
        self, dates: list, regimes: np.ndarray, obs: np.ndarray
    ) -> pl.DataFrame:
        """Package regime labels and risk params into a Polars DataFrame."""
        vt = [REGIME_VOL_TARGET[int(r)] for r in regimes]
        wc = [REGIME_WEIGHT_CAP[int(r)] for r in regimes]
        return pl.DataFrame({
            'date':        dates,
            'regime':      regimes.tolist(),
            'vol_target':  vt,
            'weight_cap':  wc,
            'obs_vol':     obs[:, 0].tolist(),
            'obs_disp':    obs[:, 1].tolist(),
            'obs_skew':    obs[:, 2].tolist(),
            'obs_kurt':    obs[:, 3].tolist(),
        })

    # ── Statistical helpers ───────────────────────────────────────────────────

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        mu, s = x.mean(), x.std()
        return float(np.mean(((x - mu) / (s + 1e-10)) ** 3))

    @staticmethod
    def _excess_kurtosis(x: np.ndarray) -> float:
        if len(x) < 4:
            return 0.0
        mu, s = x.mean(), x.std()
        return float(np.mean(((x - mu) / (s + 1e-10)) ** 4) - 3.0)
