# src/research/causality.py
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
#
# Nonlinear causality toolkit — implements the Ma & Prosperino (2023) framework.
#
# Reference:
#   Haochun Ma, Davide Prosperino, Alexander Haluszczynski, Christoph Räth,
#   "Linear and nonlinear causality in financial markets,"
#   Chaos 34, 113125 (2024). https://doi.org/10.1063/5.0184267
#
# The paper's core claim: Pearson correlation is a good proxy for LINEAR causality
# but systematically underestimates TOTAL causality by ignoring nonlinear effects.
# This module implements the three-stage decomposition proposed:
#
#   Total causality     = TE(X → Y)    or  CCM(X, Y)
#   Linear causality    = TE(X_surr → Y_surr)  [Fourier surrogates destroy nonlinearity]
#   Nonlinear causality = Total − Linear
#
# Components:
#   TransferEntropy          — Shannon-kernel TE estimator (k-nearest-neighbour)
#   FourierSurrogate         — Theiler et al. (1992) AAFT surrogate generator
#   NonlinearCausalityScore  — Decomposes TE into linear + nonlinear fractions
#   ConvergentCrossMapping   — Sugihara et al. (2012) state-space causality test
#   CausalFeatureMatrix      — Produces (N_assets × N_assets) nonlinear-causality graph
#
# Usage:
#   from causality import CausalFeatureMatrix
#   scorer = CausalFeatureMatrix(n_neighbours=5, n_surrogates=19)
#   nlc_matrix = scorer.compute(returns_df)   # pl.DataFrame (N×N)

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import polars as pl
from scipy.spatial import KDTree
from scipy.signal import periodogram


# ── Constants ─────────────────────────────────────────────────────────────────

_LOG2: Final[float] = float(np.log(2))


# ── Transfer Entropy ──────────────────────────────────────────────────────────

def _knn_entropy(X: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko k-NN entropy estimator.

    H(X) ≈ d·E[log ε_k] + log(N) − ψ(k) + log(2)
    where ε_k is the distance to the k-th nearest neighbour,
    d is the dimension, and ψ is the digamma function.

    Args:
        X: Array of shape (N, d) — N observations, d dimensions.
        k: Number of nearest neighbours (Kraskov et al. recommend k=5..10).

    Returns:
        Differential entropy estimate in nats.
    """
    from scipy.special import digamma
    N, d = X.shape
    if N <= k + 1:
        return 0.0
    tree = KDTree(X)
    # Query k+1 because the point itself is always returned at distance 0
    dists, _ = tree.query(X, k=k + 1, workers=-1)
    eps = dists[:, k]  # distance to k-th neighbour (exclude self)
    eps = np.maximum(eps, 1e-10)
    return d * np.mean(np.log(eps)) + np.log(N) - digamma(k) + np.log(2)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
    embed_dim: int = 1,
) -> float:
    """Compute Transfer Entropy TE(source → target) via k-NN estimation.

    TE(X→Y) = H(Y_t | Y_{t-1}) − H(Y_t | Y_{t-1}, X_{t-lag})
             = H(Y_t, Y_{t-1}, X_{t-lag}) − H(Y_{t-1}, X_{t-lag}) − H(Y_t, Y_{t-1}) + H(Y_{t-1})

    This is the continuous-valued generalisation using the Kraskov-Stögbauer-Grassberger
    (KSG) mutual information estimator rephrased as entropy differences.

    Args:
        source:    1-D time series array for the source variable X.
        target:    1-D time series array for the target variable Y.
        lag:       Time lag (τ) for the source influence (default: 1 day).
        k:         k-NN parameter for entropy estimation.
        embed_dim: Embedding dimension for Y history (Takens theorem; default 1).

    Returns:
        Transfer entropy in nats (≥ 0; values near 0 indicate no information flow).
    """
    N = min(len(source), len(target))
    # Build joint arrays with lag alignment
    # Y_t, Y_{t-1}, X_{t-lag} — all must be contemporaneous after slicing
    start = max(embed_dim, lag)
    end   = N

    yt      = target[start:end].reshape(-1, 1)
    yt_past = np.column_stack([target[start - j : end - j] for j in range(1, embed_dim + 1)])
    x_lag   = source[start - lag : end - lag].reshape(-1, 1)

    # Joint spaces for entropy decomposition
    joint_all  = np.concatenate([yt, yt_past, x_lag], axis=1)   # (N', d+2)
    joint_yx   = np.concatenate([yt, yt_past],         axis=1)   # (N', d+1)
    joint_past = np.concatenate([yt_past, x_lag],      axis=1)   # (N', d+1)

    H_all  = _knn_entropy(joint_all,  k)
    H_yx   = _knn_entropy(joint_yx,   k)
    H_past = _knn_entropy(joint_past, k)
    H_yp   = _knn_entropy(yt_past,    k)

    # TE = H(Y_t, Y_{t-1}) + H(Y_{t-1}, X_{t-lag}) − H(Y_t, Y_{t-1}, X_{t-lag}) − H(Y_{t-1})
    te = H_yx + H_past - H_all - H_yp
    return max(te, 0.0)


# ── Fourier Surrogate ─────────────────────────────────────────────────────────

def fourier_aaft_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplitude-Adjusted Fourier Transform (AAFT) surrogate.

    Proposed by Theiler et al. (1992). The surrogate preserves the LINEAR
    autocorrelation structure (power spectrum) and marginal distribution
    of x, but DESTROYS nonlinear dependencies.

    By comparing TE(original) vs mean(TE(surrogates)), we isolate the
    nonlinear contribution as Ma & Prosperino (2023) propose.

    Algorithm:
        1. Rank-sort x → r (Gaussian reference)
        2. FFT of r → R_f
        3. Randomise phases of R_f while preserving conjugate symmetry
        4. IFFT → r_surr
        5. Rank-match r_surr back to x's amplitude distribution

    Args:
        x:   1-D time series.
        rng: numpy Generator for reproducible results.

    Returns:
        Surrogate array with same shape and dtype as x.
    """
    N = len(x)
    # Step 1: Gaussian reference series via rank transform
    rank  = np.argsort(np.argsort(x))
    gauss = rng.standard_normal(N)
    gauss_sorted = np.sort(gauss)
    r = gauss_sorted[rank]

    # Step 2: FFT of Gaussian reference
    R_f = np.fft.rfft(r)

    # Step 3: Random phase rotation (preserves power spectrum)
    n_freq = len(R_f)
    phases = rng.uniform(0, 2 * np.pi, n_freq)
    R_f_surr = np.abs(R_f) * np.exp(1j * phases)

    # Step 4: IFFT back to time domain
    r_surr = np.fft.irfft(R_f_surr, n=N)

    # Step 5: Rank-match to original amplitude distribution
    rank_surr = np.argsort(np.argsort(r_surr))
    x_sorted  = np.sort(x)
    return x_sorted[rank_surr].astype(x.dtype)


# ── Nonlinear Causality Score ─────────────────────────────────────────────────

@dataclass
class NonlinearCausalityResult:
    """Decomposed causality result for a single directed pair X → Y.

    Attributes:
        te_total:      Raw TE(X → Y) on original data.
        te_linear:     Mean TE over AAFT surrogates (≈ linear contribution).
        te_nonlinear:  te_total − te_linear (nonlinear surplus).
        nl_fraction:   te_nonlinear / te_total (0 = purely linear, 1 = purely nonlinear).
        p_value:       Fraction of surrogate TEs ≥ te_total (one-sided rank test).
        significant:   True if p_value < significance_level.
    """
    te_total:     float
    te_linear:    float
    te_nonlinear: float
    nl_fraction:  float
    p_value:      float
    significant:  bool


def nonlinear_causality_score(
    source:             np.ndarray,
    target:             np.ndarray,
    n_surrogates:       int = 19,
    lag:                int = 1,
    k:                  int = 5,
    significance_level: float = 0.05,
    seed:               int = 42,
) -> NonlinearCausalityResult:
    """Decomposes TE(source → target) into linear and nonlinear contributions.

    Follows the framework of Ma & Prosperino (2023):
        1. Compute TE on original data (total causality).
        2. Generate n_surrogates AAFT surrogates for both series.
        3. Compute TE on each surrogate pair (linear causality proxy).
        4. Nonlinear causality = original TE − mean surrogate TE.
        5. p-value: proportion of surrogates with TE ≥ original TE.

    The minimum n_surrogates for significance level α is ⌈1/α⌉ − 1.
    For α=0.05, n_surrogates=19 gives exact rank-based test.

    Args:
        source:             1-D time series for the source.
        target:             1-D time series for the target.
        n_surrogates:       Number of AAFT surrogates (default: 19).
        lag:                TE lag in bars (default: 1).
        k:                  k-NN parameter for entropy estimation.
        significance_level: Threshold for significance test.
        seed:               RNG seed for surrogate generation.

    Returns:
        NonlinearCausalityResult with full decomposition.
    """
    rng = np.random.default_rng(seed)

    te_orig = transfer_entropy(source, target, lag=lag, k=k)

    surr_tes = np.empty(n_surrogates)
    for i in range(n_surrogates):
        src_s = fourier_aaft_surrogate(source, rng)
        tgt_s = fourier_aaft_surrogate(target, rng)
        surr_tes[i] = transfer_entropy(src_s, tgt_s, lag=lag, k=k)

    te_linear    = float(surr_tes.mean())
    te_nonlinear = max(te_orig - te_linear, 0.0)
    nl_fraction  = te_nonlinear / (te_orig + 1e-10)
    p_value      = float(np.mean(surr_tes >= te_orig))

    return NonlinearCausalityResult(
        te_total     = te_orig,
        te_linear    = te_linear,
        te_nonlinear = te_nonlinear,
        nl_fraction  = nl_fraction,
        p_value      = p_value,
        significant  = p_value < significance_level,
    )


# ── Convergent Cross-Mapping ──────────────────────────────────────────────────

def convergent_cross_mapping(
    x:          np.ndarray,
    y:          np.ndarray,
    embed_dim:  int = 3,
    tau:        int = 1,
    lib_sizes:  list[int] | None = None,
) -> dict[str, float]:
    """Sugihara et al. (2012) Convergent Cross-Mapping (CCM) causality test.

    CCM tests whether X causally influences Y by checking if the state-space
    of Y can be used to predict X. This detects causality in WEAKLY COUPLED
    dynamical systems where correlation may be near zero but causality exists —
    exactly the nonlinear causality missed by Pearson correlation.

    The key signature of causation: CCM skill INCREASES as library size grows
    (convergence), as opposed to spurious correlation which is library-size-independent.

    Algorithm:
        1. Embed Y into a delay-coordinate manifold M_Y using embed_dim and tau.
        2. For each point in M_Y, find its nearest neighbours.
        3. Use those neighbours to predict X (cross-map).
        4. Report Pearson ρ(X_actual, X_predicted) at multiple library sizes.

    Args:
        x:          1-D causal source (the variable we test as driver).
        y:          1-D causal target (the manifold we reconstruct on).
        embed_dim:  Takens embedding dimension (choose via FNN or AMI).
        tau:        Time delay for embedding (choose via first AMI minimum).
        lib_sizes:  Library sizes at which to evaluate convergence.

    Returns:
        Dict with keys 'rho_max' (skill at full library), 'convergence_slope',
        and 'rho_by_lib' (list of (lib_size, rho) pairs).
    """
    N = min(len(x), len(y))
    if lib_sizes is None:
        lib_sizes = [int(N * f) for f in [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]]
        lib_sizes = sorted(set(max(embed_dim + 2, l) for l in lib_sizes))

    # Delay-coordinate embedding of Y → manifold M_Y
    # M_Y[t] = [y[t], y[t-tau], y[t-2*tau], ..., y[t-(E-1)*tau]]
    max_lag = (embed_dim - 1) * tau
    M_len   = N - max_lag
    M_Y     = np.column_stack([
        y[max_lag - j * tau: N - j * tau] for j in range(embed_dim)
    ])  # shape (M_len, embed_dim)

    # Aligned x (contemporaneous with M_Y rows)
    x_aligned = x[max_lag:N]

    rho_by_lib: list[tuple[int, float]] = []

    for lib_size in lib_sizes:
        lib_size = min(lib_size, M_len)
        lib_M    = M_Y[:lib_size]
        lib_x    = x_aligned[:lib_size]

        # For each point in FULL manifold, find E+1 nearest neighbours in library
        tree = KDTree(lib_M)
        dists, idxs = tree.query(M_Y, k=embed_dim + 1, workers=-1)

        # Weighted prediction: w_i = exp(-d_i / d_1) normalised
        eps       = np.maximum(dists[:, 0:1], 1e-10)
        weights   = np.exp(-dists / eps)
        weights  /= weights.sum(axis=1, keepdims=True)

        # Clip indices to library
        idxs_clipped = np.clip(idxs, 0, lib_size - 1)
        x_pred = (weights * lib_x[idxs_clipped]).sum(axis=1)

        if np.std(x_pred) < 1e-10 or np.std(x_aligned) < 1e-10:
            rho = 0.0
        else:
            rho = float(np.corrcoef(x_aligned, x_pred)[0, 1])

        rho_by_lib.append((lib_size, rho))

    rhos = [r for _, r in rho_by_lib]
    # Convergence slope: positive → evidence of causation
    if len(rhos) >= 2:
        lib_ns = np.array([l for l, _ in rho_by_lib], dtype=float)
        slope  = float(np.polyfit(lib_ns / lib_ns.max(), rhos, 1)[0])
    else:
        slope = 0.0

    return {
        'rho_max':          max(rhos) if rhos else 0.0,
        'convergence_slope': slope,
        'rho_by_lib':       rho_by_lib,
    }


# ── Causal Feature Matrix ─────────────────────────────────────────────────────

@dataclass
class CausalFeatureMatrix:
    """Produces an (N_assets × N_assets) pairwise nonlinear causality graph.

    For each ordered pair (i, j), computes:
        - TE_nonlinear(i → j)  — how much nonlinear info flows from asset i to j
        - CCM ρ_max(i → j)     — Sugihara convergence-based causality score

    These become features of the cross-asset alpha model. Assets that CAUSE
    others nonlinearly (high out-degree in the causal graph) tend to be leading
    indicators; assets that are CAUSED (high in-degree) are followers.

    Attributes:
        n_neighbours:     k-NN parameter for TE estimation.
        n_surrogates:     AAFT surrogates for linear/nonlinear decomposition.
        lag:              Time lag for TE (in bars).
        min_history:      Minimum number of bars required per asset.
        use_ccm:          Whether to also compute CCM (slower; richer signal).
    """

    n_neighbours: int   = 5
    n_surrogates: int   = 19
    lag:          int   = 1
    min_history:  int   = 126
    use_ccm:      bool  = True

    def compute(
        self,
        returns_df: pl.DataFrame,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Compute pairwise TE and CCM causality matrices.

        Args:
            returns_df: Wide DataFrame — rows=dates, columns=asset returns.
            seed:       RNG seed for AAFT surrogates.

        Returns:
            Tuple of (te_nonlinear_df, ccm_rho_df), both N_assets × N_assets.
            Diagonal entries are NaN (self-causality undefined).
        """
        assets = [c for c in returns_df.columns if c != 'date']
        N = len(assets)
        ret_np = returns_df.select(assets).to_numpy().astype(np.float64)

        te_mat   = np.full((N, N), np.nan)
        ccm_mat  = np.full((N, N), np.nan)

        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i == j:
                    continue
                x = ret_np[:, i]
                y = ret_np[:, j]

                # Drop NaNs
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < self.min_history:
                    continue
                x_c, y_c = x[mask], y[mask]

                try:
                    result = nonlinear_causality_score(
                        x_c, y_c,
                        n_surrogates=self.n_surrogates,
                        lag=self.lag,
                        k=self.n_neighbours,
                        seed=seed + i * N + j,
                    )
                    te_mat[i, j] = result.te_nonlinear if result.significant else 0.0
                except Exception:
                    te_mat[i, j] = 0.0

                if self.use_ccm:
                    try:
                        ccm_res = convergent_cross_mapping(x_c, y_c)
                        ccm_mat[i, j] = ccm_res['rho_max'] * max(ccm_res['convergence_slope'], 0.0)
                    except Exception:
                        ccm_mat[i, j] = 0.0

        te_df  = pl.DataFrame({a: te_mat[:, j]  for j, a in enumerate(assets)}).with_columns(
            pl.Series('asset', assets)
        )
        ccm_df = pl.DataFrame({a: ccm_mat[:, j] for j, a in enumerate(assets)}).with_columns(
            pl.Series('asset', assets)
        )
        return te_df, ccm_df

    def compute_asset_causal_scores(
        self,
        returns_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Summarises the causal graph into per-asset scalar features.

        Returns a DataFrame with one row per asset and columns:
            - f_te_out_degree:  Sum of nonlinear TE flowing OUT of this asset.
              A high value → this asset LEADS others nonlinearly (alpha source).
            - f_te_in_degree:   Sum of nonlinear TE flowing INTO this asset.
              A high value → this asset FOLLOWS others (potentially crowded).
            - f_te_net_cause:   out_degree − in_degree (signed causality centrality).
            - f_ccm_driver:     Max CCM ρ for paths originating from this asset.
        """
        te_df, ccm_df = self.compute(returns_df)
        assets = [c for c in te_df.columns if c != 'asset']

        te_mat  = te_df.select(assets).to_numpy()
        ccm_mat = ccm_df.select(assets).to_numpy()

        # Replace NaN with 0 for aggregation
        te_mat  = np.nan_to_num(te_mat,  nan=0.0)
        ccm_mat = np.nan_to_num(ccm_mat, nan=0.0)

        out_degree = te_mat.sum(axis=1)   # row i = sum of TE(i → j)
        in_degree  = te_mat.sum(axis=0)   # col j = sum of TE(i → j)
        net_cause  = out_degree - in_degree
        ccm_driver = ccm_mat.max(axis=1)

        return pl.DataFrame({
            'asset':            assets,
            'f_te_out_degree':  out_degree.tolist(),
            'f_te_in_degree':   in_degree.tolist(),
            'f_te_net_cause':   net_cause.tolist(),
            'f_ccm_driver':     ccm_driver.tolist(),
        })
