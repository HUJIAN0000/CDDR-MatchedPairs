# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta2s_ultrafast_cpu.py

Purpose
-------
Ultra-fast CPU implementation of the 5% matched-pair joint analysis. 
It combines 38 galaxy-cluster angular-diameter distances and their matched 
Pantheon+ SNe Ia with four DESI BAO D_M/r_d measurements, while treating r_d 
as a free parameter. The affine-invariant ensemble sampler and likelihood are 
compiled with Numba; the statistical model is unchanged.

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
NDIM = 4
NWALKERS = 32
NSTEPS = 60000
BURN_IN = 15000
STRETCH_A = 2.0
MAX_CLUSTER_CACHE = 4096
RANDOM_SEED = None

LOG_2PI = np.log(2.0 * np.pi)
LN10_OVER_5 = np.log(10.0) / 5.0
FIVE_OVER_LN10 = 5.0 / np.log(10.0)
HASH_MULTIPLIER = np.uint64(11400714819323198485)


# =============================================================================
# 2. Data loading
# =============================================================================
def load_all_data():
    print("Loading galaxy-cluster data...")
    df = pd.read_csv("matched_result_cluster.csv")

    z_cl = np.ascontiguousarray(df["z"].to_numpy(dtype=np.float64))
    dA = np.ascontiguousarray(df["DA(GPC)"].to_numpy(dtype=np.float64) * 1000.0)
    sp_mpc = np.ascontiguousarray(df["SDA+"].to_numpy(dtype=np.float64) * 1000.0)
    sm_mpc = np.ascontiguousarray(df["SDA-"].to_numpy(dtype=np.float64) * 1000.0)
    mb_cl = np.ascontiguousarray(
        df["sn_matched_m_b_corr"].to_numpy(dtype=np.float64)
    )

    if z_cl.size > 64:
        raise ValueError(
            f"The ultra-fast 64-bit cache supports at most 64 galaxy-cluster data points; received {z_cl.size}. "
            "Use test_cluster_M_dm_eta2s_fast_cpu.py instead."
        )

    sp_mu = FIVE_OVER_LN10 * (sp_mpc / dA)
    sm_mu = FIVE_OVER_LN10 * (sm_mpc / dA)
    var_plus = np.ascontiguousarray(sp_mu * sp_mu)
    var_minus = np.ascontiguousarray(sm_mu * sm_mu)

    cov_cl = np.loadtxt("cov_cluster_lens.txt", dtype=np.float64)
    n_cl = z_cl.size
    if cov_cl.shape != (n_cl, n_cl):
        raise ValueError(
            f"cov_cluster_lens.txt shape {cov_cl.shape} does not match the sample size {n_cl}."
        )
    if not np.allclose(cov_cl, cov_cl.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The galaxy-cluster covariance matrix is slightly asymmetric and has been symmetrized.")
        cov_cl = 0.5 * (cov_cl + cov_cl.T)
    cov_cl = np.ascontiguousarray(cov_cl)

    mu_base_cl = 5.0 * np.log10(dA * (1.0 + z_cl) ** 2) + 25.0
    diff_base_cl = np.ascontiguousarray(mb_cl - mu_base_cl)

    print("Loading DESI BAO measurements and the corresponding Pantheon+ SNe Ia...")
    z_bao = np.ascontiguousarray(
        np.array([0.510, 0.706, 0.930, 1.317], dtype=np.float64)
    )
    obs_bao = np.ascontiguousarray(
        np.array([13.62003, 16.84645, 21.70841, 27.78720], dtype=np.float64)
    )
    bao_var = np.ascontiguousarray(
        np.array([0.06346622, 0.10197571, 0.07956752, 0.47656986], dtype=np.float64)
    )
    mb_bao = np.ascontiguousarray(
        np.array([22.8415, 23.6432, 24.2801, 25.5337], dtype=np.float64)
    )
    cov_sn_bao = np.loadtxt("cov_desi_sn.txt", dtype=np.float64)

    n_bao = z_bao.size
    if cov_sn_bao.shape != (n_bao, n_bao):
        raise ValueError(
            f"cov_desi_sn.txt shape {cov_sn_bao.shape} does not match the number of BAO measurements {n_bao}."
        )
    if not np.allclose(cov_sn_bao, cov_sn_bao.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The Pantheon+ sub-covariance matrix for the DESI anchors is slightly asymmetric and has been symmetrized.")
        cov_sn_bao = 0.5 * (cov_sn_bao + cov_sn_bao.T)
    cov_sn_bao = np.ascontiguousarray(cov_sn_bao)

    one_plus_z_bao = np.ascontiguousarray(1.0 + z_bao)
    dl_prefactor_bao = np.ascontiguousarray(
        np.exp(LN10_OVER_5 * (mb_bao - 25.0))
    )

    cluster_data = (
        z_cl, diff_base_cl, var_plus, var_minus, cov_cl,
        dA, sp_mu, sm_mu, mb_cl,
    )
    bao_data = (
        z_bao, one_plus_z_bao, obs_bao, bao_var,
        dl_prefactor_bao, cov_sn_bao, mb_bao,
    )
    print(f"✅ Loaded {n_cl} galaxy-cluster points and {n_bao} BAO points successfully")
    return cluster_data, bao_data


# =============================================================================
# 3. Numba-compiled cache and joint likelihood
# =============================================================================
@njit(inline="always", cache=True)
def _cache_lookup(key, table_keys, table_values):
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
def _joint_log_prob_one(
    theta,
    z_cl,
    diff_base_cl,
    var_plus_cl,
    var_minus_cl,
    cov_cl,
    z_bao,
    one_plus_z_bao,
    obs_bao,
    bao_var,
    dl_prefactor_bao,
    cov_sn_bao,
    diff_cl,
    matrix_cl,
    diff_bao,
    model_bao,
    matrix_bao,
    table_keys,
    table_values,
    cached_linv,
    cached_log_norm,
    cache_count,
    cache_stats,
):
    M = theta[0]
    dm = theta[1]
    eta = theta[2]
    rd = theta[3]

    if not (-20.5 < M < -18.0):
        return -np.inf, cache_count
    if not (-1.0 < dm < 1.0):
        return -np.inf, cache_count
    if not (-1.0 < eta < 1.0):
        return -np.inf, cache_count
    if not (100.0 < rd < 200.0):
        return -np.inf, cache_count

    n_cl = z_cl.size
    key = np.uint64(0)
    for i in range(n_cl):
        eta_term = 1.0 + eta * z_cl[i]
        if eta_term <= 0.0:
            return -np.inf, cache_count
        value = (
            diff_base_cl[i] - M - dm * z_cl[i]
            - 5.0 * np.log10(eta_term)
        )
        diff_cl[i] = value
        if value > 0.0:
            key |= np.uint64(1) << np.uint64(i)

    slot, hash_position = _cache_lookup(key, table_keys, table_values)
    cluster_log_like = -np.inf

    if slot >= 0:
        cache_stats[0] += 1
        chi2_cl = 0.0
        for i in range(n_cl):
            y_i = 0.0
            for j in range(i + 1):
                y_i += cached_linv[slot, i, j] * diff_cl[j]
            chi2_cl += y_i * y_i
        cluster_log_like = -0.5 * (chi2_cl + cached_log_norm[slot])
    else:
        cache_stats[1] += 1
        for i in range(n_cl):
            for j in range(n_cl):
                matrix_cl[i, j] = cov_cl[i, j]
            if diff_cl[i] > 0.0:
                matrix_cl[i, i] += var_minus_cl[i]
            else:
                matrix_cl[i, i] += var_plus_cl[i]

        log_det_cl = 0.0
        for i in range(n_cl):
            for j in range(i + 1):
                value = matrix_cl[i, j]
                for k in range(j):
                    value -= matrix_cl[i, k] * matrix_cl[j, k]
                if i == j:
                    if value <= 0.0:
                        return -np.inf, cache_count
                    value = np.sqrt(value)
                    matrix_cl[i, j] = value
                    log_det_cl += 2.0 * np.log(value)
                else:
                    matrix_cl[i, j] = value / matrix_cl[j, j]

        log_norm_cl = log_det_cl + n_cl * LOG_2PI

        if cache_count < cached_linv.shape[0]:
            slot = cache_count
            cache_count += 1
            table_keys[hash_position] = key
            table_values[hash_position] = slot
            cached_log_norm[slot] = log_norm_cl

            for i in range(n_cl):
                for j in range(n_cl):
                    cached_linv[slot, i, j] = 0.0

            for col in range(n_cl):
                cached_linv[slot, col, col] = 1.0 / matrix_cl[col, col]
                for i in range(col + 1, n_cl):
                    value = 0.0
                    for k in range(col, i):
                        value += matrix_cl[i, k] * cached_linv[slot, k, col]
                    cached_linv[slot, i, col] = -value / matrix_cl[i, i]

            chi2_cl = 0.0
            for i in range(n_cl):
                y_i = 0.0
                for j in range(i + 1):
                    y_i += cached_linv[slot, i, j] * diff_cl[j]
                chi2_cl += y_i * y_i
        else:
            cache_stats[2] += 1
            chi2_cl = 0.0
            for i in range(n_cl):
                value = diff_cl[i]
                for k in range(i):
                    value -= matrix_cl[i, k] * diff_cl[k]
                value /= matrix_cl[i, i]
                diff_cl[i] = value
                chi2_cl += value * value

        cluster_log_like = -0.5 * (chi2_cl + log_norm_cl)

    # -------------------- DESI BAO 4 x 4 block --------------------
    n_bao = z_bao.size
    for i in range(n_bao):
        eta_term = 1.0 + eta * z_bao[i]
        if eta_term <= 0.0:
            return -np.inf, cache_count

        DL = dl_prefactor_bao[i] * np.exp(
            -LN10_OVER_5 * (M + dm * z_bao[i])
        )
        model = DL / (one_plus_z_bao[i] * eta_term * rd)
        model_bao[i] = model
        diff_bao[i] = obs_bao[i] - model

    for i in range(n_bao):
        ji = LN10_OVER_5 * model_bao[i]
        for j in range(n_bao):
            jj = LN10_OVER_5 * model_bao[j]
            matrix_bao[i, j] = cov_sn_bao[i, j] * ji * jj
        matrix_bao[i, i] += bao_var[i]

    log_det_bao = 0.0
    for i in range(n_bao):
        for j in range(i + 1):
            value = matrix_bao[i, j]
            for k in range(j):
                value -= matrix_bao[i, k] * matrix_bao[j, k]
            if i == j:
                if value <= 0.0:
                    return -np.inf, cache_count
                value = np.sqrt(value)
                matrix_bao[i, j] = value
                log_det_bao += 2.0 * np.log(value)
            else:
                matrix_bao[i, j] = value / matrix_bao[j, j]

    chi2_bao = 0.0
    for i in range(n_bao):
        value = diff_bao[i]
        for k in range(i):
            value -= matrix_bao[i, k] * diff_bao[k]
        value /= matrix_bao[i, i]
        diff_bao[i] = value
        chi2_bao += value * value

    bao_log_like = -0.5 * (
        chi2_bao + log_det_bao + n_bao * LOG_2PI
    )
    return cluster_log_like + bao_log_like, cache_count


# =============================================================================
# 4. Fully compiled Stretch-Move sampler
# =============================================================================
@njit(cache=True)
def _run_stretch_sampler_joint(
    initial_pos,
    nsteps,
    stretch_a,
    seed,
    z_cl,
    diff_base_cl,
    var_plus_cl,
    var_minus_cl,
    cov_cl,
    z_bao,
    one_plus_z_bao,
    obs_bao,
    bao_var,
    dl_prefactor_bao,
    cov_sn_bao,
    max_cache,
):
    np.random.seed(seed)

    pos = initial_pos.copy()
    nwalkers, ndim = pos.shape
    half = nwalkers // 2
    n_cl = z_cl.size
    n_bao = z_bao.size

    table_size = 1
    while table_size < 4 * max_cache:
        table_size *= 2

    table_keys = np.zeros(table_size, dtype=np.uint64)
    table_values = np.full(table_size, -1, dtype=np.int64)
    cached_linv = np.empty((max_cache, n_cl, n_cl), dtype=np.float64)
    cached_log_norm = np.empty(max_cache, dtype=np.float64)
    cache_count = 0
    cache_stats = np.zeros(3, dtype=np.int64)

    diff_cl = np.empty(n_cl, dtype=np.float64)
    matrix_cl = np.empty((n_cl, n_cl), dtype=np.float64)
    diff_bao = np.empty(n_bao, dtype=np.float64)
    model_bao = np.empty(n_bao, dtype=np.float64)
    matrix_bao = np.empty((n_bao, n_bao), dtype=np.float64)

    current_log_prob = np.empty(nwalkers, dtype=np.float64)
    for i in range(nwalkers):
        current_log_prob[i], cache_count = _joint_log_prob_one(
            pos[i],
            z_cl, diff_base_cl, var_plus_cl, var_minus_cl, cov_cl,
            z_bao, one_plus_z_bao, obs_bao, bao_var,
            dl_prefactor_bao, cov_sn_bao,
            diff_cl, matrix_cl, diff_bao, model_bao, matrix_bao,
            table_keys, table_values, cached_linv, cached_log_norm,
            cache_count, cache_stats,
        )

    chain = np.empty((nsteps, nwalkers, ndim), dtype=np.float64)
    accepted = np.zeros(nwalkers, dtype=np.int64)
    permutation = np.arange(nwalkers)
    proposal = np.empty(ndim, dtype=np.float64)

    for step in range(nsteps):
        np.random.shuffle(permutation)
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

                new_log_prob, cache_count = _joint_log_prob_one(
                    proposal,
                    z_cl, diff_base_cl, var_plus_cl, var_minus_cl, cov_cl,
                    z_bao, one_plus_z_bao, obs_bao, bao_var,
                    dl_prefactor_bao, cov_sn_bao,
                    diff_cl, matrix_cl, diff_bao, model_bao, matrix_bao,
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
# 5. Final single-sample reference likelihood
# =============================================================================
def ln_likelihood_reference(theta, cluster_data, bao_data):
    M, dm, eta, rd = np.asarray(theta, dtype=np.float64)
    z_cl, _, _, _, cov_cl, dA, sp_mu, sm_mu, mb_cl = cluster_data
    z_bao, one_plus_z_bao, obs_bao, bao_var, _, cov_sn_bao, mb_bao = bao_data

    eta_cl = 1.0 + eta * z_cl
    eta_bao = 1.0 + eta * z_bao
    if np.any(eta_cl <= 0.0) or np.any(eta_bao <= 0.0) or rd <= 0.0:
        return -np.inf

    mu_cl = 5.0 * np.log10(dA * (1.0 + z_cl) ** 2 * eta_cl) + 25.0
    mu_sn_cl = mb_cl - (M + dm * z_cl)
    diff_cl = mu_sn_cl - mu_cl
    sigmas = np.where(diff_cl > 0.0, sm_mu, sp_mu)
    cov_cl_total = cov_cl.copy()
    cov_cl_total[np.diag_indices(z_cl.size)] += sigmas * sigmas

    try:
        L_cl = scipy.linalg.cholesky(
            cov_cl_total, lower=True, check_finite=False
        )
        y_cl = scipy.linalg.solve_triangular(
            L_cl, diff_cl, lower=True, check_finite=False
        )
    except scipy.linalg.LinAlgError:
        return -np.inf

    ll_cl = -0.5 * (
        float(y_cl @ y_cl)
        + 2.0 * np.log(np.diag(L_cl)).sum()
        + z_cl.size * LOG_2PI
    )

    mu_sn_bao = mb_bao - (M + dm * z_bao)
    DL = 10.0 ** ((mu_sn_bao - 25.0) / 5.0)
    model = DL / (one_plus_z_bao * eta_bao * rd)
    diff_bao = obs_bao - model
    jdiag = LN10_OVER_5 * model
    cov_bao = cov_sn_bao * (jdiag[:, None] * jdiag[None, :])
    cov_bao = cov_bao.copy()
    cov_bao[np.diag_indices(z_bao.size)] += bao_var

    try:
        L_bao = scipy.linalg.cholesky(cov_bao, lower=True, check_finite=False)
        y_bao = scipy.linalg.solve_triangular(
            L_bao, diff_bao, lower=True, check_finite=False
        )
    except scipy.linalg.LinAlgError:
        return -np.inf

    ll_bao = -0.5 * (
        float(y_bao @ y_bao)
        + 2.0 * np.log(np.diag(L_bao)).sum()
        + z_bao.size * LOG_2PI
    )
    return ll_cl + ll_bao


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

    cluster_data, bao_data = load_all_data()
    (
        z_cl, diff_base_cl, var_plus_cl, var_minus_cl, cov_cl,
        _, _, _, _,
    ) = cluster_data
    (
        z_bao, one_plus_z_bao, obs_bao, bao_var,
        dl_prefactor_bao, cov_sn_bao, _,
    ) = bao_data

    seed = _make_seed()
    rng = np.random.default_rng(seed)
    initial = np.array([-19.25, 0.0, 0.0, 147.1], dtype=np.float64)
    initial_pos = initial + 1.0e-3 * rng.standard_normal((NWALKERS, NDIM))

    estimated_cache_mib = MAX_CLUSTER_CACHE * z_cl.size * z_cl.size * 8 / 1024**2
    print(f"Cache capacity: {MAX_CLUSTER_CACHE} residual-sign patterns; approximately {estimated_cache_mib:.1f} MiB")
    print(f"Random seed: {seed}")

    print("Loading/compiling Numba kernels...")
    compile_start = time.perf_counter()
    _run_stretch_sampler_joint(
        initial_pos, 1, STRETCH_A, seed,
        z_cl, diff_base_cl, var_plus_cl, var_minus_cl, cov_cl,
        z_bao, one_plus_z_bao, obs_bao, bao_var,
        dl_prefactor_bao, cov_sn_bao,
        MAX_CLUSTER_CACHE,
    )
    compile_elapsed = time.perf_counter() - compile_start
    print(f"✅ Numba kernels ready: {compile_elapsed:.2f} s")

    print(f"\n🚀 Starting ultra-fast joint CPU MCMC: {NSTEPS} steps × {NWALKERS} walkers")
    start = time.perf_counter()
    chain, accepted, final_log_prob, cache_count, cache_stats = (
        _run_stretch_sampler_joint(
            initial_pos, NSTEPS, STRETCH_A, seed,
            z_cl, diff_base_cl, var_plus_cl, var_minus_cl, cov_cl,
            z_bao, one_plus_z_bao, obs_bao, bao_var,
            dl_prefactor_bao, cov_sn_bao,
            MAX_CLUSTER_CACHE,
        )
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
    print("\n⚡ Galaxy-cluster covariance-cache statistics")
    print(f"   Residual-sign patterns in cache : {cache_count:,}")
    print(f"   Cache hits     : {cache_stats[0]:,}")
    print(f"   Cholesky factorizations    : {cache_stats[1]:,}")
    print(f"   Uncached evaluations after cache saturation : {cache_stats[2]:,}")

    flat_samples = np.ascontiguousarray(chain[BURN_IN:].reshape(-1, NDIM))
    labels = [r"M_0", r"\varepsilon", r"\eta", r"r_d"]

    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    best_params = np.array([s["median"] for s in stats], dtype=np.float64)
    lnL_best = ln_likelihood_reference(best_params, cluster_data, bao_data)

    cosmo_tools.print_results(
        stats,
        lnL_best=lnL_best,
        num_data=z_cl.size + z_bao.size,
    )

    cosmo_tools.plot_getdist_advanced(
        flat_samples,
        labels,
        stats,
        filename="Pantheon_Joint_Result_GetDists_ultrafast_cpu.pdf",
    )


if __name__ == "__main__":
    main()
