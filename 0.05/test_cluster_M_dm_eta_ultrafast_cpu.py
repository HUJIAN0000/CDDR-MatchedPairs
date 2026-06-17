# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta_ultrafast_cpu.py

Purpose
-------
Ultra-fast CPU implementation of the cluster-only 5% matched-pair CDDR analysis
 using 38 galaxy-cluster angular-diameter distances and matched Pantheon+ 
 SNe Ia. A Numba-compiled affine-invariant ensemble sampler jointly constrains
 M_0, epsilon, and eta while retaining the asymmetric cluster-distance likelihood.

The code variable 'dm' denotes the linear SNe Ia absolute-magnitude evolution 
coefficient epsilon in M_B(z) = M_0 + epsilon z.
Associated manuscript: 'A model-independent test of the cosmic distance-duality
 relation using galaxy clusters and Type Ia supernovae matched pairs'.

@author:Jian Hu
Email:dg1626002@smail.nju.edu.cn
'''

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import scipy.linalg
from numba import njit

import cosmo_tools


# =============================================================================
# 1. Configuration
# =============================================================================
NDIM = 3
NWALKERS = 32
NSTEPS = 60000
BURN_IN = 15000
STRETCH_A = 2.0

# For 38 data points, each cache entry uses about 11.3 KiB; 4096 entries require about 45 MiB.
MAX_CLUSTER_CACHE = 4096
RANDOM_SEED = None

LOG_2PI = np.log(2.0 * np.pi)
FIVE_OVER_LN10 = 5.0 / np.log(10.0)
HASH_MULTIPLIER = np.uint64(11400714819323198485)


# =============================================================================
# 2. Data loading and precomputation
# =============================================================================
def load_data():
    print("Loading input data...")
    df = pd.read_csv("matched_result_cluster.csv")

    z = np.ascontiguousarray(df["z"].to_numpy(dtype=np.float64))
    dA = np.ascontiguousarray(df["DA(GPC)"].to_numpy(dtype=np.float64) * 1000.0)
    sda_plus = np.ascontiguousarray(df["SDA+"].to_numpy(dtype=np.float64) * 1000.0)
    sda_minus = np.ascontiguousarray(df["SDA-"].to_numpy(dtype=np.float64) * 1000.0)
    mb = np.ascontiguousarray(df["sn_matched_m_b_corr"].to_numpy(dtype=np.float64))

    sig_plus = FIVE_OVER_LN10 * (sda_plus / dA)
    sig_minus = FIVE_OVER_LN10 * (sda_minus / dA)
    var_plus = np.ascontiguousarray(sig_plus * sig_plus)
    var_minus = np.ascontiguousarray(sig_minus * sig_minus)

    cov = np.loadtxt("cov_cluster_lens.txt", dtype=np.float64)
    n = z.size
    if n > 64:
        raise ValueError(
            f"The ultra-fast 64-bit cache supports at most 64 galaxy-cluster data points; received {n}. "
            "Use test_cluster_M_dm_eta_fast_cpu.py instead."
        )
    if cov.shape != (n, n):
        raise ValueError(f"Covariance-matrix shape {cov.shape} does not match the sample size {n}.")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The covariance matrix is slightly asymmetric and has been symmetrized.")
        cov = 0.5 * (cov + cov.T)
    cov = np.ascontiguousarray(cov)

    # diff = diff_base - M - dm*z - 5log10(1+eta*z)
    mu_base = 5.0 * np.log10(dA * (1.0 + z) ** 2) + 25.0
    diff_base = np.ascontiguousarray(mb - mu_base)

    print(f"✅ Data loaded successfully; sample size: {n}")
    return z, diff_base, var_plus, var_minus, cov, dA, sig_plus, sig_minus, mb


# =============================================================================
# 3. Numba-compiled hash cache and single-sample likelihood
# =============================================================================
@njit(inline="always", cache=True)
def _cache_lookup(key, table_keys, table_values):
    """Open-addressing hash table: return (cache slot or -1, hash position available for insertion)."""
    mask = table_values.size - 1
    h = np.int64((key * HASH_MULTIPLIER) & np.uint64(mask))
    while True:
        slot = table_values[h]
        if slot == -1:
            return -1, h
        if table_keys[h] == key:
            return slot, h
        h = (h + 1) & mask


@njit(inline="always", cache=True)
def _cluster_log_prob_one(
    theta,
    z,
    diff_base,
    var_plus,
    var_minus,
    cov,
    diff_scratch,
    matrix_scratch,
    table_keys,
    table_values,
    cached_linv,
    cached_log_norm,
    cache_count,
    cache_stats,
):
    """Return (log posterior, updated cache_count); flat-prior bounds are checked here."""
    M = theta[0]
    dm = theta[1]
    eta = theta[2]

    if not (-20.5 < M < -18.0):
        return -np.inf, cache_count
    if not (-1.0 < dm < 1.0):
        return -np.inf, cache_count
    if not (-1.0 < eta < 1.0):
        return -np.inf, cache_count

    n = z.size
    key = np.uint64(0)

    for i in range(n):
        eta_term = 1.0 + eta * z[i]
        if eta_term <= 0.0:
            return -np.inf, cache_count
        value = diff_base[i] - M - dm * z[i] - 5.0 * np.log10(eta_term)
        diff_scratch[i] = value
        if value > 0.0:
            key |= np.uint64(1) << np.uint64(i)

    slot, hash_position = _cache_lookup(key, table_keys, table_values)

    # -------------------- Cache hit: O(N^2) whitening only --------------------
    if slot >= 0:
        cache_stats[0] += 1  # hit
        chi2 = 0.0
        for i in range(n):
            y_i = 0.0
            for j in range(i + 1):
                y_i += cached_linv[slot, i, j] * diff_scratch[j]
            chi2 += y_i * y_i
        return -0.5 * (chi2 + cached_log_norm[slot]), cache_count

    # -------------------- Cache miss: construct C and compute its Cholesky factor --------------------
    cache_stats[1] += 1  # miss/factorization
    for i in range(n):
        for j in range(n):
            matrix_scratch[i, j] = cov[i, j]
        if diff_scratch[i] > 0.0:
            matrix_scratch[i, i] += var_minus[i]
        else:
            matrix_scratch[i, i] += var_plus[i]

    log_det = 0.0
    # In-place lower-triangular Cholesky factorization: C -> L
    for i in range(n):
        for j in range(i + 1):
            value = matrix_scratch[i, j]
            for k in range(j):
                value -= matrix_scratch[i, k] * matrix_scratch[j, k]

            if i == j:
                if value <= 0.0:
                    return -np.inf, cache_count
                value = np.sqrt(value)
                matrix_scratch[i, j] = value
                log_det += 2.0 * np.log(value)
            else:
                matrix_scratch[i, j] = value / matrix_scratch[j, j]

    log_norm = log_det + n * LOG_2PI

    # -------------------- Cache available: store L^{-1} --------------------
    if cache_count < cached_linv.shape[0]:
        slot = cache_count
        cache_count += 1
        table_keys[hash_position] = key
        table_values[hash_position] = slot
        cached_log_norm[slot] = log_norm

        for i in range(n):
            for j in range(n):
                cached_linv[slot, i, j] = 0.0

        # Compute the lower-triangular inverse L^{-1} column by column
        for col in range(n):
            cached_linv[slot, col, col] = 1.0 / matrix_scratch[col, col]
            for i in range(col + 1, n):
                value = 0.0
                for k in range(col, i):
                    value += matrix_scratch[i, k] * cached_linv[slot, k, col]
                cached_linv[slot, i, col] = -value / matrix_scratch[i, i]

        chi2 = 0.0
        for i in range(n):
            y_i = 0.0
            for j in range(i + 1):
                y_i += cached_linv[slot, i, j] * diff_scratch[j]
            chi2 += y_i * y_i
        return -0.5 * (chi2 + log_norm), cache_count

    # -------------------- Cache full: use direct forward substitution --------------------
    cache_stats[2] += 1
    chi2 = 0.0
    for i in range(n):
        value = diff_scratch[i]
        for k in range(i):
            value -= matrix_scratch[i, k] * diff_scratch[k]
        value /= matrix_scratch[i, i]
        diff_scratch[i] = value
        chi2 += value * value

    return -0.5 * (chi2 + log_norm), cache_count


# =============================================================================
# 4. Numba-compiled affine-invariant ensemble sampler
# =============================================================================
@njit(cache=True)
def _run_stretch_sampler(
    initial_pos,
    nsteps,
    stretch_a,
    seed,
    z,
    diff_base,
    var_plus,
    var_minus,
    cov,
    max_cache,
):
    np.random.seed(seed)

    pos = initial_pos.copy()
    nwalkers, ndim = pos.shape
    half = nwalkers // 2
    n = z.size

    # Keep a low hash-table load factor by using a power-of-two size >= 4*max_cache.
    table_size = 1
    while table_size < 4 * max_cache:
        table_size *= 2

    table_keys = np.zeros(table_size, dtype=np.uint64)
    table_values = np.full(table_size, -1, dtype=np.int64)
    cached_linv = np.empty((max_cache, n, n), dtype=np.float64)
    cached_log_norm = np.empty(max_cache, dtype=np.float64)
    cache_count = 0
    # [hits, misses/factorizations, cache_full_uncached]
    cache_stats = np.zeros(3, dtype=np.int64)

    diff_scratch = np.empty(n, dtype=np.float64)
    matrix_scratch = np.empty((n, n), dtype=np.float64)

    current_log_prob = np.empty(nwalkers, dtype=np.float64)
    for i in range(nwalkers):
        current_log_prob[i], cache_count = _cluster_log_prob_one(
            pos[i], z, diff_base, var_plus, var_minus, cov,
            diff_scratch, matrix_scratch,
            table_keys, table_values, cached_linv, cached_log_norm,
            cache_count, cache_stats,
        )

    chain = np.empty((nsteps, nwalkers, ndim), dtype=np.float64)
    accepted = np.zeros(nwalkers, dtype=np.int64)
    permutation = np.arange(nwalkers)
    proposal = np.empty(ndim, dtype=np.float64)

    for step in range(nsteps):
        np.random.shuffle(permutation)

        # Update the two complementary subsets sequentially, following the red-blue StretchMove structure.
        for split in range(2):
            for local_i in range(half):
                if split == 0:
                    walker_i = permutation[local_i]
                    partner_i = permutation[half + np.random.randint(half)]
                else:
                    walker_i = permutation[half + local_i]
                    partner_i = permutation[np.random.randint(half)]

                u = np.random.random()
                z_scale = ((stretch_a - 1.0) * u + 1.0) ** 2 / stretch_a

                for d in range(ndim):
                    proposal[d] = (
                        pos[partner_i, d]
                        + z_scale * (pos[walker_i, d] - pos[partner_i, d])
                    )

                new_log_prob, cache_count = _cluster_log_prob_one(
                    proposal, z, diff_base, var_plus, var_minus, cov,
                    diff_scratch, matrix_scratch,
                    table_keys, table_values, cached_linv, cached_log_norm,
                    cache_count, cache_stats,
                )

                log_accept_ratio = (
                    (ndim - 1.0) * np.log(z_scale)
                    + new_log_prob
                    - current_log_prob[walker_i]
                )
                if np.log(np.random.random()) < log_accept_ratio:
                    for d in range(ndim):
                        pos[walker_i, d] = proposal[d]
                    current_log_prob[walker_i] = new_log_prob
                    accepted[walker_i] += 1

        for i in range(nwalkers):
            for d in range(ndim):
                chain[step, i, d] = pos[i, d]

    return chain, accepted, current_log_prob, cache_count, cache_stats


# =============================================================================
# 5. Single-sample reference likelihood, evaluated once for AIC/BIC
# =============================================================================
def ln_likelihood_reference(theta, z, dA, sig_plus, sig_minus, mb, cov):
    M, dm, eta = np.asarray(theta, dtype=np.float64)
    eta_term = 1.0 + eta * z
    if np.any(eta_term <= 0.0):
        return -np.inf

    dL = dA * (1.0 + z) ** 2 * eta_term
    mu_cluster = 5.0 * np.log10(dL) + 25.0
    mu_sn = mb - (M + dm * z)
    diff = mu_sn - mu_cluster

    sigmas = np.where(diff > 0.0, sig_minus, sig_plus)
    cov_total = cov.copy()
    cov_total[np.diag_indices(z.size)] += sigmas * sigmas

    try:
        L = scipy.linalg.cholesky(cov_total, lower=True, check_finite=False)
        y = scipy.linalg.solve_triangular(L, diff, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        return -np.inf

    return -0.5 * (
        float(y @ y)
        + 2.0 * np.log(np.diag(L)).sum()
        + z.size * LOG_2PI
    )


def _make_seed():
    if RANDOM_SEED is not None:
        return int(RANDOM_SEED)
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0] % 2147483647)


def main():
    if NWALKERS % 2 != 0:
        raise ValueError("NWALKERS must be even")
    if NWALKERS < 2 * NDIM:
        raise ValueError("NWALKERS must be at least 2*NDIM")
    if not (0 <= BURN_IN < NSTEPS):
        raise ValueError("BURN_IN must satisfy 0 <= BURN_IN < NSTEPS")

    z, diff_base, var_plus, var_minus, cov, dA, sig_plus, sig_minus, mb = load_data()

    seed = _make_seed()
    rng = np.random.default_rng(seed)
    initial = np.array([-19.25, 0.0, 0.0], dtype=np.float64)
    initial_pos = initial + 1.0e-3 * rng.standard_normal((NWALKERS, NDIM))

    estimated_cache_mib = MAX_CLUSTER_CACHE * z.size * z.size * 8 / 1024**2
    print(f"Cache capacity: {MAX_CLUSTER_CACHE} residual-sign patterns; approximately {estimated_cache_mib:.1f} MiB")
    print(f"Random seed: {seed}")

    # The first call triggers JIT compilation; copies and an independent call preserve the production chain.
    print("Loading/compiling Numba kernels...")
    compile_start = time.perf_counter()
    _run_stretch_sampler(
        initial_pos,
        1,
        STRETCH_A,
        seed,
        z,
        diff_base,
        var_plus,
        var_minus,
        cov,
        MAX_CLUSTER_CACHE,
    )
    compile_elapsed = time.perf_counter() - compile_start
    print(f"✅ Numba kernels ready: {compile_elapsed:.2f} s")

    print(f"\n🚀 Starting ultra-fast CPU MCMC: {NSTEPS} steps × {NWALKERS} walkers")
    start = time.perf_counter()
    chain, accepted, final_log_prob, cache_count, cache_stats = _run_stretch_sampler(
        initial_pos,
        NSTEPS,
        STRETCH_A,
        seed,
        z,
        diff_base,
        var_plus,
        var_minus,
        cov,
        MAX_CLUSTER_CACHE,
    )
    elapsed = time.perf_counter() - start

    acceptance = accepted / float(NSTEPS)
    total_evals = NWALKERS * (NSTEPS + 1)
    print(f"\n✅ MCMC completed; production-sampling time: {elapsed:.3f} s")
    print(f"   Sampling rate: {NSTEPS / elapsed:,.1f} steps/s")
    print(f"   Likelihood throughput: {total_evals / elapsed:,.0f} evaluations/s")
    print(
        f"   Acceptance fraction: mean={acceptance.mean():.3f}, "
        f"min={acceptance.min():.3f}, max={acceptance.max():.3f}"
    )
    print("\n⚡ Covariance-cache statistics")
    print(f"   Residual-sign patterns in cache : {cache_count:,}")
    print(f"   Cache hits     : {cache_stats[0]:,}")
    print(f"   Cholesky factorizations    : {cache_stats[1]:,}")
    print(f"   Uncached evaluations after cache saturation : {cache_stats[2]:,}")

    flat_samples = np.ascontiguousarray(chain[BURN_IN:].reshape(-1, NDIM))
    labels = [r"M_0", r"\varepsilon", r"\eta"]

    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    best_params = np.array([s["median"] for s in stats], dtype=np.float64)
    lnL_best = ln_likelihood_reference(
        best_params, z, dA, sig_plus, sig_minus, mb, cov
    )

    cosmo_tools.print_results(stats, lnL_best=lnL_best, num_data=z.size)

    cosmo_tools.plot_getdist_advanced(
        flat_samples,
        labels,
        stats,
        filename="Cluster_Result_GetDist1_ultrafast_cpu.pdf",
    )


if __name__ == "__main__":
    main()
