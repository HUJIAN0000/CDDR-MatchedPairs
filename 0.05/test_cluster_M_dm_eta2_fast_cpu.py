# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta2_fast_cpu.py

Purpose
-------
Vectorized CPU implementation of the 5% matched-pair joint analysis. 
It combines 38 galaxy-cluster angular-diameter distances and their matched 
Pantheon+ SNe Ia with four DESI BAO D_M/r_d measurements and the Gaussian 
Planck prior r_d = 147.09 +/- 0.26 Mpc. Cached covariance factorizations 
and batched linear algebra accelerate the emcee likelihood evaluation without 
changing the statistical model.

The code variable 'dm' denotes the linear SNe Ia absolute-magnitude evolution 
coefficient epsilon in M_B(z) = M_0 + epsilon z.
Associated manuscript: 'A model-independent test of the cosmic distance-duality
 relation using galaxy clusters and Type Ia supernovae matched pairs'.

@author:Jian Hu
Email:dg1626002@smail.nju.edu.cn
'''

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import time

import emcee
import numpy as np
import pandas as pd
import scipy.linalg

import cosmo_tools

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


# =============================================================================
# 1. Configuration
# =============================================================================
NDIM = 4
NWALKERS = 32
NSTEPS = 60000
BURN_IN = 15000
BLAS_THREADS = 1
MAX_CLUSTER_CACHE = 8192
RANDOM_SEED = None
PROGRESS = True

LOG_2PI = np.log(2.0 * np.pi)
LN10_OVER_5 = np.log(10.0) / 5.0
PLANCK_RD_MEAN = 147.09
PLANCK_RD_SIGMA = 0.26
FIVE_OVER_LN10 = 5.0 / np.log(10.0)


# =============================================================================
# 2. Data loading and preprocessing
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

    sp_mu = FIVE_OVER_LN10 * (sp_mpc / dA)
    sm_mu = FIVE_OVER_LN10 * (sm_mpc / dA)
    cov_cl = np.loadtxt("cov_cluster_lens.txt", dtype=np.float64)

    n_cl = z_cl.size
    if cov_cl.shape != (n_cl, n_cl):
        raise ValueError(
            f"cov_cluster_lens.txt shape {cov_cl.shape} does not match the galaxy-cluster sample size {n_cl}."
        )
    if not np.allclose(cov_cl, cov_cl.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The galaxy-cluster covariance matrix is slightly asymmetric and has been symmetrized.")
        cov_cl = 0.5 * (cov_cl + cov_cl.T)
    cov_cl = np.asfortranarray(cov_cl)

    print("Loading DESI BAO measurements and the corresponding Pantheon+ SNe Ia...")
    z_bao = np.ascontiguousarray(
        np.array([0.510, 0.706, 0.930, 1.317], dtype=np.float64)
    )
    dm_over_rd_obs = np.ascontiguousarray(
        np.array([13.62003, 16.84645, 21.70841, 27.78720], dtype=np.float64)
    )
    bao_var = np.ascontiguousarray(
        np.array([0.06346622, 0.10197571, 0.07956752, 0.47656986], dtype=np.float64)
    )
    mb_desi = np.ascontiguousarray(
        np.array([22.8415, 23.6432, 24.2801, 25.5337], dtype=np.float64)
    )
    cov_desi_sn = np.loadtxt("cov_desi_sn.txt", dtype=np.float64)

    n_bao = z_bao.size
    if cov_desi_sn.shape != (n_bao, n_bao):
        raise ValueError(
            f"cov_desi_sn.txt shape {cov_desi_sn.shape} does not match the number of BAO measurements {n_bao}."
        )
    if not np.allclose(cov_desi_sn, cov_desi_sn.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The Pantheon+ sub-covariance matrix for the DESI anchors is slightly asymmetric and has been symmetrized.")
        cov_desi_sn = 0.5 * (cov_desi_sn + cov_desi_sn.T)
    cov_desi_sn = np.ascontiguousarray(cov_desi_sn)

    data_cl = (z_cl, dA, sp_mu, sm_mu, mb_cl, cov_cl)
    data_bao = (z_bao, dm_over_rd_obs, bao_var, mb_desi, cov_desi_sn)
    return data_cl, data_bao


# =============================================================================
# 3. Optimized joint likelihood
# =============================================================================
class FastJointLogProb:
    def __init__(self, data_cl, data_bao, max_cache=MAX_CLUSTER_CACHE):
        z_cl, dA, sp_mu, sm_mu, mb_cl, cov_cl = data_cl
        z_bao, obs_bao, bao_var, mb_bao, cov_sn_bao = data_bao

        # Galaxy-cluster contribution
        self.z_cl = z_cl
        self.n_cl = z_cl.size
        self.cov_cl = cov_cl
        self.diag_cl = np.diag_indices(self.n_cl)
        self.eye_cl = np.eye(self.n_cl, dtype=np.float64, order="F")
        mu_base_cl = 5.0 * np.log10(dA * (1.0 + z_cl) ** 2) + 25.0
        self.diff_base_cl = np.ascontiguousarray(mb_cl - mu_base_cl)
        self.var_plus_cl = np.ascontiguousarray(sp_mu * sp_mu)
        self.var_minus_cl = np.ascontiguousarray(sm_mu * sm_mu)

        # DESI BAO contribution
        self.z_bao = z_bao
        self.n_bao = z_bao.size
        self.obs_bao = obs_bao
        self.bao_var = bao_var
        self.cov_sn_bao = cov_sn_bao
        self.one_plus_z_bao = 1.0 + z_bao
        self.dl_prefactor_bao = np.exp(LN10_OVER_5 * (mb_bao - 25.0))
        self.diag_bao = np.arange(self.n_bao)

        self.max_cache = int(max_cache)
        self.cache: OrderedDict[bytes, tuple[np.ndarray, float]] = OrderedDict()
        self.cache_requests = 0
        self.cache_hits = 0
        self.factorizations = 0
        self.walker_evaluations = 0

    def _cluster_whitener(self, key, minus_mask):
        self.cache_requests += 1
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return cached

        cov_tot = self.cov_cl.copy(order="F")
        cov_tot[self.diag_cl] += np.where(
            minus_mask, self.var_minus_cl, self.var_plus_cl
        )

        try:
            L = scipy.linalg.cholesky(
                cov_tot,
                lower=True,
                overwrite_a=True,
                check_finite=False,
            )
        except scipy.linalg.LinAlgError:
            return np.empty((0, 0), dtype=np.float64), -np.inf

        log_norm = 2.0 * np.log(np.diag(L)).sum() + self.n_cl * LOG_2PI
        L_inv = scipy.linalg.solve_triangular(
            L,
            self.eye_cl,
            lower=True,
            check_finite=False,
            overwrite_b=False,
        )
        whitener_t = np.ascontiguousarray(L_inv.T)
        result = (whitener_t, float(log_norm))
        self.factorizations += 1

        if self.max_cache > 0:
            self.cache[key] = result
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        return result

    def _cluster_loglike_batch(self, M, dm, eta):
        batch = M.size
        out = np.full(batch, -np.inf, dtype=np.float64)

        eta_term = 1.0 + eta[:, None] * self.z_cl[None, :]
        domain_ok = np.all(eta_term > 0.0, axis=1)
        if not np.any(domain_ok):
            return out

        rows = np.flatnonzero(domain_ok)
        diff = (
            self.diff_base_cl[None, :]
            - M[rows, None]
            - dm[rows, None] * self.z_cl[None, :]
            - 5.0 * np.log10(eta_term[rows])
        )

        minus_masks = diff > 0.0
        packed = np.packbits(minus_masks, axis=1, bitorder="little")
        groups = defaultdict(list)
        for local_i in range(diff.shape[0]):
            groups[packed[local_i].tobytes()].append(local_i)

        ll = np.full(diff.shape[0], -np.inf, dtype=np.float64)
        for key, local_rows_list in groups.items():
            local_rows = np.asarray(local_rows_list, dtype=np.intp)
            whitener_t, log_norm = self._cluster_whitener(
                key, minus_masks[local_rows[0]]
            )
            if not np.isfinite(log_norm):
                continue
            whitened = diff[local_rows] @ whitener_t
            chi2 = np.einsum("ij,ij->i", whitened, whitened, optimize=True)
            ll[local_rows] = -0.5 * (chi2 + log_norm)

        out[rows] = ll
        return out

    def _bao_loglike_batch(self, M, dm, eta, rd):
        batch = M.size
        out = np.full(batch, -np.inf, dtype=np.float64)

        eta_term = 1.0 + eta[:, None] * self.z_bao[None, :]
        domain_ok = np.all(eta_term > 0.0, axis=1) & (rd > 0.0)
        if not np.any(domain_ok):
            return out

        rows = np.flatnonzero(domain_ok)
        Mv = M[rows, None]
        dmv = dm[rows, None]
        etav_term = eta_term[rows]
        rdv = rd[rows, None]

        # Rewrite D_L = 10**((m_B-M_0-epsilon*z-25)/5) as a precomputed factor multiplied by an exponential.
        DL = self.dl_prefactor_bao[None, :] * np.exp(
            -LN10_OVER_5 * (Mv + dmv * self.z_bao[None, :])
        )
        model = DL / (self.one_plus_z_bao[None, :] * etav_term * rdv)
        diff = self.obs_bao[None, :] - model

        # Because J is diagonal, evaluate J C J^T by element-wise scaling without constructing np.diag(J).
        jdiag = LN10_OVER_5 * model
        cov_batch = (
            self.cov_sn_bao[None, :, :]
            * jdiag[:, :, None]
            * jdiag[:, None, :]
        )
        cov_batch[:, self.diag_bao, self.diag_bao] += self.bao_var[None, :]

        try:
            # NumPy supports batched Cholesky factorization for stacked matrices.
            L = np.linalg.cholesky(cov_batch)
            # Solve L y = diff and use chi2 = y^T y, avoiding an explicit C^{-1} diff operation.
            y = np.linalg.solve(L, diff[..., None])[..., 0]
            chi2 = np.einsum("ij,ij->i", y, y, optimize=True)
            log_norm = (
                2.0
                * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
                + self.n_bao * LOG_2PI
            )
            out[rows] = -0.5 * (chi2 + log_norm)
        except np.linalg.LinAlgError:
            # In rare cases, process walkers individually so that one non-positive-definite matrix does not invalidate the full batch.
            for local_i, global_i in enumerate(rows):
                try:
                    Li = np.linalg.cholesky(cov_batch[local_i])
                    yi = np.linalg.solve(Li, diff[local_i])
                    chi2_i = float(yi @ yi)
                    log_norm_i = (
                        2.0 * np.log(np.diag(Li)).sum() + self.n_bao * LOG_2PI
                    )
                    out[global_i] = -0.5 * (chi2_i + log_norm_i)
                except np.linalg.LinAlgError:
                    out[global_i] = -np.inf

        return out

    def log_likelihood_batch(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]

        M, dm, eta, rd = theta.T
        ll_cl = self._cluster_loglike_batch(M, dm, eta)
        ll_bao = self._bao_loglike_batch(M, dm, eta, rd)
        self.walker_evaluations += theta.shape[0]

        # Include the same Gaussian Planck prior on r_d as in the reference joint analysis.
        chi2_planck = ((rd - PLANCK_RD_MEAN) / PLANCK_RD_SIGMA) ** 2
        out = ll_cl + ll_bao - 0.5 * chi2_planck
        out[~np.isfinite(out)] = -np.inf
        return out

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        scalar_input = theta.ndim == 1
        if scalar_input:
            theta = theta[None, :]

        M, dm, eta, rd = theta.T
        prior_ok = (
            (M > -20.5)
            & (M < -18.0)
            & (dm > -1.0)
            & (dm < 1.0)
            & (eta > -1.0)
            & (eta < 1.0)
            & (rd > 130.0)
            & (rd < 160.0)
        )

        out = np.full(theta.shape[0], -np.inf, dtype=np.float64)
        if np.any(prior_ok):
            idx = np.flatnonzero(prior_ok)
            out[idx] = self.log_likelihood_batch(theta[idx])

        return out[0] if scalar_input else out

    def log_likelihood(self, theta):
        return float(self.log_likelihood_batch(np.asarray(theta)[None, :])[0])

    def print_cache_stats(self):
        hit_rate = (
            100.0 * self.cache_hits / self.cache_requests
            if self.cache_requests
            else 0.0
        )
        print("\n⚡ Galaxy-cluster covariance-cache statistics")
        print(f"   Joint walker likelihood evaluations : {self.walker_evaluations:,}")
        print(f"   Cholesky factorizations performed  : {self.factorizations:,}")
        print(f"   Current number of cached patterns      : {len(self.cache):,}")
        print(f"   Pattern-request cache hit rate  : {hit_rate:.2f}%")


# =============================================================================
# 4. Main program
# =============================================================================
def main():
    data_cl, data_bao = load_all_data()
    n_data = data_cl[0].size + data_bao[0].size + 1  # +1: Planck prior on r_d

    fast_log_prob = FastJointLogProb(
        data_cl, data_bao, max_cache=MAX_CLUSTER_CACHE
    )

    print("\n🚀 Starting vectorized CPU joint MCMC for (M_0, epsilon, eta, r_d)...")
    rng = np.random.default_rng(RANDOM_SEED)
    initial = np.array([-19.25, 0.0, 0.0, 147.1], dtype=np.float64)
    pos = initial + 1.0e-3 * rng.standard_normal((NWALKERS, NDIM))

    sampler = emcee.EnsembleSampler(
        NWALKERS,
        NDIM,
        fast_log_prob,
        vectorize=True,
    )

    thread_ctx = (
        threadpool_limits(limits=BLAS_THREADS)
        if threadpool_limits is not None
        else nullcontext()
    )

    start_time = time.perf_counter()
    with thread_ctx:
        sampler.run_mcmc(pos, NSTEPS, progress=PROGRESS)
    elapsed = time.perf_counter() - start_time

    print(f"\n✅ MCMC completed; elapsed time: {elapsed:.2f} s")
    print(f"   Mean sampling rate: {NSTEPS / elapsed:.2f} steps/s")
    fast_log_prob.print_cache_stats()

    flat_samples = sampler.get_chain(discard=BURN_IN, thin=1, flat=True)
    labels = [r"M_0", r"\varepsilon", r"\eta", r"r_d"]

    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    best_params = np.array([s["median"] for s in stats], dtype=np.float64)
    lnL_best = fast_log_prob.log_likelihood(best_params)

    cosmo_tools.print_results(stats, lnL_best=lnL_best, num_data=n_data)
    cosmo_tools.plot_getdist_advanced(
        flat_samples,
        labels,
        stats,
        filename="Pantheon_Joint_Result_GetDist_fast_cpu.pdf",
    )


if __name__ == "__main__":
    main()
