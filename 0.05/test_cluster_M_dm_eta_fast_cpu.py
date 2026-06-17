# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta_fast_cpu.py

Purpose
-------
Vectorized CPU implementation of the cluster-only 5% matched-pair CDDR analysis 
using 38 galaxy-cluster angular-diameter distances and matched Pantheon+ SNe Ia. 
Cached covariance factorizations accelerate the emcee likelihood evaluation 
while preserving the asymmetric cluster-distance uncertainties.

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
import sys
import time

import emcee
import numpy as np
import pandas as pd
import scipy.linalg

import cosmo_tools

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # threadpoolctl is commonly installed with the SciPy environment
    threadpool_limits = None


# =============================================================================
# 1. Configuration
# =============================================================================
NDIM = 3
NWALKERS = 32
NSTEPS = 60000
BURN_IN = 15000

# For matrices of order a few tens, single-threaded BLAS usually avoids thread-launch overhead.
BLAS_THREADS = 1

# Each cache entry uses approximately 8*N*N bytes; for N=38, 8192 entries require about 90 MiB, excluding Python overhead.
# Reduce this to 2048/4096 on memory-limited systems, or increase it to 16384 when memory permits.
MAX_CLUSTER_CACHE = 8192

RANDOM_SEED = None
PROGRESS = True

LOG_2PI = np.log(2.0 * np.pi)
FIVE_OVER_LN10 = 5.0 / np.log(10.0)


# =============================================================================
# 2. Data loading
# =============================================================================
def load_data():
    print("Loading input data...")
    try:
        df = pd.read_csv("matched_result_cluster.csv")
    except FileNotFoundError:
        print("❌ Error: file not found: matched_result_cluster.csv")
        sys.exit(1)

    z = np.ascontiguousarray(df["z"].to_numpy(dtype=np.float64))
    dA = np.ascontiguousarray(df["DA(GPC)"].to_numpy(dtype=np.float64) * 1000.0)
    sda_plus = np.ascontiguousarray(df["SDA+"].to_numpy(dtype=np.float64) * 1000.0)
    sda_minus = np.ascontiguousarray(df["SDA-"].to_numpy(dtype=np.float64) * 1000.0)
    mb = np.ascontiguousarray(df["sn_matched_m_b_corr"].to_numpy(dtype=np.float64))

    sig_plus = FIVE_OVER_LN10 * (sda_plus / dA)
    sig_minus = FIVE_OVER_LN10 * (sda_minus / dA)

    try:
        cov = np.loadtxt("cov_cluster_lens.txt", dtype=np.float64)
    except FileNotFoundError:
        print("❌ Error: file not found: cov_cluster_lens.txt")
        sys.exit(1)

    n = z.size
    if cov.shape != (n, n):
        print(f"❌ Covariance-matrix shape {cov.shape} does not match the number of data points {n}.")
        sys.exit(1)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        print("⚠️ The covariance matrix is slightly asymmetric and has been symmetrized as (C + C.T)/2.")
        cov = 0.5 * (cov + cov.T)

    cov = np.asfortranarray(cov)
    print(f"✅ Data loaded successfully; sample size: {n}")
    return z, dA, sig_plus, sig_minus, mb, cov


# =============================================================================
# 3. Vectorized likelihood
# =============================================================================
class FastClusterLogProb:
    """Evaluate log probability in batches and cache the whitening matrix for each asymmetric-error sign pattern."""

    def __init__(
        self,
        z: np.ndarray,
        dA: np.ndarray,
        sig_plus: np.ndarray,
        sig_minus: np.ndarray,
        mb: np.ndarray,
        cov: np.ndarray,
        max_cache: int = MAX_CLUSTER_CACHE,
    ) -> None:
        self.z = z
        self.n = z.size
        self.cov = cov
        self.max_cache = int(max_cache)
        self.diag_idx = np.diag_indices(self.n)
        self.identity = np.eye(self.n, dtype=np.float64, order="F")

        # Original expression:
        # diff = mb - M - dm*z - [5log10(dA*(1+z)^2*(1+eta*z)) + 25]
        # All parameter-independent terms are precomputed.
        mu_base = 5.0 * np.log10(dA * (1.0 + z) ** 2) + 25.0
        self.diff_base = np.ascontiguousarray(mb - mu_base)
        self.var_plus = np.ascontiguousarray(sig_plus * sig_plus)
        self.var_minus = np.ascontiguousarray(sig_minus * sig_minus)

        # key -> (L^{-T}, log(det(C)) + N*log(2*pi))
        self.cache: OrderedDict[bytes, tuple[np.ndarray, float]] = OrderedDict()
        self.cache_requests = 0
        self.cache_hits = 0
        self.factorizations = 0
        self.walker_evaluations = 0

    def _build_or_get_whitener(
        self, key: bytes, minus_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Return L^{-T}; compute a Cholesky factorization only when the pattern is absent from the cache."""
        self.cache_requests += 1
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return cached

        cov_tot = self.cov.copy(order="F")
        added_var = np.where(minus_mask, self.var_minus, self.var_plus)
        cov_tot[self.diag_idx] += added_var

        try:
            # C = L L^T
            L = scipy.linalg.cholesky(
                cov_tot,
                lower=True,
                overwrite_a=True,
                check_finite=False,
            )
        except scipy.linalg.LinAlgError:
            # This should not occur for valid data; a failed factorization assigns -inf to the affected group.
            result = (np.empty((0, 0), dtype=np.float64), -np.inf)
            return result

        log_norm = 2.0 * np.log(np.diag(L)).sum() + self.n * LOG_2PI

        # Reused covariance patterns avoid repeated triangular solves:
        # For row vectors, y = L^{-1} diff is evaluated as y_row = diff_row @ L^{-T}.
        L_inv = scipy.linalg.solve_triangular(
            L,
            self.identity,
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

    def cluster_loglike_batch(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate only the likelihood; theta has shape (batch, 3)."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]

        M = theta[:, 0]
        dm = theta[:, 1]
        eta = theta[:, 2]
        batch = theta.shape[0]
        out = np.full(batch, -np.inf, dtype=np.float64)

        # Enforce the log10 domain because the eta prior alone does not guarantee 1 + eta*z > 0.
        eta_term = 1.0 + eta[:, None] * self.z[None, :]
        domain_ok = np.all(eta_term > 0.0, axis=1)
        if not np.any(domain_ok):
            return out

        valid_rows = np.flatnonzero(domain_ok)
        Mv = M[valid_rows]
        dmv = dm[valid_rows]
        eta_term_v = eta_term[valid_rows]

        diff = (
            self.diff_base[None, :]
            - Mv[:, None]
            - dmv[:, None] * self.z[None, :]
            - 5.0 * np.log10(eta_term_v)
        )

        minus_masks = diff > 0.0  # As in the reference implementation, diff > 0 selects sig_mu_minus
        packed = np.packbits(minus_masks, axis=1, bitorder="little")

        # Group walkers with identical residual-sign patterns and evaluate each group with one matrix operation.
        groups: dict[bytes, list[int]] = defaultdict(list)
        for local_i in range(diff.shape[0]):
            groups[packed[local_i].tobytes()].append(local_i)

        ll_valid = np.full(diff.shape[0], -np.inf, dtype=np.float64)
        for key, local_rows_list in groups.items():
            local_rows = np.asarray(local_rows_list, dtype=np.intp)
            whitener_t, log_norm = self._build_or_get_whitener(
                key, minus_masks[local_rows[0]]
            )
            if not np.isfinite(log_norm):
                continue

            whitened = diff[local_rows] @ whitener_t
            chi2 = np.einsum("ij,ij->i", whitened, whitened, optimize=True)
            ll_valid[local_rows] = -0.5 * (chi2 + log_norm)

        self.walker_evaluations += diff.shape[0]
        out[valid_rows] = ll_valid
        return out

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Vectorized interface required by emcee; returns a batch of log-posterior values."""
        theta = np.asarray(theta, dtype=np.float64)
        scalar_input = theta.ndim == 1
        if scalar_input:
            theta = theta[None, :]

        M = theta[:, 0]
        dm = theta[:, 1]
        eta = theta[:, 2]
        prior_ok = (
            (M > -20.5)
            & (M < -18.0)
            & (dm > -1.0)
            & (dm < 1.0)
            & (eta > -1.0)
            & (eta < 1.0)
        )

        out = np.full(theta.shape[0], -np.inf, dtype=np.float64)
        if np.any(prior_ok):
            idx = np.flatnonzero(prior_ok)
            out[idx] = self.cluster_loglike_batch(theta[idx])

        return out[0] if scalar_input else out

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Evaluate the likelihood for one parameter vector for AIC/BIC calculations."""
        return float(self.cluster_loglike_batch(np.asarray(theta)[None, :])[0])

    def print_cache_stats(self) -> None:
        hit_rate = (
            100.0 * self.cache_hits / self.cache_requests
            if self.cache_requests
            else 0.0
        )
        print("\n⚡ Covariance-cache statistics")
        print(f"   Walker likelihood evaluations : {self.walker_evaluations:,}")
        print(f"   Cholesky factorizations performed  : {self.factorizations:,}")
        print(f"   Current number of cached patterns      : {len(self.cache):,}")
        print(f"   Pattern-request cache hit rate  : {hit_rate:.2f}%")


# =============================================================================
# 4. Main program
# =============================================================================
def main() -> None:
    z, dA, sig_plus, sig_minus, mb, cov = load_data()
    n = z.size

    fast_log_prob = FastClusterLogProb(
        z, dA, sig_plus, sig_minus, mb, cov, max_cache=MAX_CLUSTER_CACHE
    )

    print("\n🚀 Starting vectorized CPU MCMC analysis for (M_0, epsilon, eta)...")
    rng = np.random.default_rng(RANDOM_SEED)
    initial = np.array([-19.25, 0.0, 0.0], dtype=np.float64)
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
    labels = [r"M_0", r"\varepsilon", r"\eta"]

    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    best_params = np.array([s["median"] for s in stats], dtype=np.float64)
    lnL_best = fast_log_prob.log_likelihood(best_params)

    cosmo_tools.print_results(stats, lnL_best=lnL_best, num_data=n)

    cosmo_tools.plot_getdist_advanced(
        flat_samples,
        labels,
        stats,
        filename="Cluster_Result_GetDist1_fast_cpu.pdf",
    )


if __name__ == "__main__":
    main()
