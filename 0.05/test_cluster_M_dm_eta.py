# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta.py

Purpose
-------
Cluster-only 5% matched-pair CDDR analysis using 38 galaxy-cluster 
angular-diameter distances and matched Pantheon+ SNe Ia. The script 
jointly constrains M_0, epsilon, and the CDDR-deformation parameter eta, 
propagating the Pantheon+ sub-covariance matrix and asymmetric cluster-distance 
uncertainties.

The code variable 'dm' denotes the linear SNe Ia absolute-magnitude evolution 
coefficient epsilon in M_B(z) = M_0 + epsilon z.
Associated manuscript: 'A model-independent test of the cosmic distance-duality 
relation using galaxy clusters and Type Ia supernovae matched pairs'.

@author:Jian Hu
Email:dg1626002@smail.nju.edu.cn
'''

import numpy as np
import pandas as pd
import scipy.linalg
import sys, os
import emcee
import time
from multiprocessing import Pool

# Import the shared post-processing utilities
import cosmo_tools

# =============================================================================
# 1. Configuration
# =============================================================================
ndim = 3          # Parameters: [M_0, epsilon, eta]
nwalkers = 32
nsteps = 4000     
burn_in = 1000 
use_multiprocessing = True 

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

    z_cluster = df['z'].values
    dA_Mpc = df['DA(GPC)'].values * 1000.0 
    SdA_plus_Mpc = df['SDA+'].values * 1000.0
    SdA_minus_Mpc = df['SDA-'].values * 1000.0

    # Precompute distance-modulus uncertainties
    COEFF = 5.0 / np.log(10.0)
    sig_mu_plus = COEFF * (SdA_plus_Mpc / dA_Mpc)
    sig_mu_minus = COEFF * (SdA_minus_Mpc / dA_Mpc)

    mb_corr = df['sn_matched_m_b_corr'].values
    
    try:
        cov_sn = np.loadtxt("cov_cluster_lens.txt")
    except FileNotFoundError:
        print("❌ Error: file not found: cov_cluster_lens.txt")
        sys.exit(1)
        
    N = len(z_cluster)
    if cov_sn.shape != (N, N):
        print(f"❌ Covariance-matrix shape {cov_sn.shape} does not match the number of data points {N}.")
        sys.exit(1)
        
    print(f"✅ Data loaded successfully; sample size: {N}")
    return z_cluster, dA_Mpc, sig_mu_plus, sig_mu_minus, mb_corr, cov_sn, N

# =============================================================================
# 3. Likelihood function
# =============================================================================
def ln_likelihood(theta, z_cluster, dA_Mpc, sig_mu_plus, sig_mu_minus, mb_corr, cov_sn):
    M, dm, eta = theta
    
    # Theoretical prediction
    term_z = (1 + z_cluster)**2 *(1+ eta*z_cluster)
    # Modified CDDR luminosity-distance relation
    dL_model = dA_Mpc * term_z
    mu_cluster_model = 5.0 * np.log10(dL_model) + 25.0
    mu_sn_model = mb_corr - (M + dm * z_cluster)
    diff = mu_sn_model - mu_cluster_model
    
    # Piecewise treatment of asymmetric cluster uncertainties
    cluster_sigmas = np.where(diff > 0, sig_mu_minus, sig_mu_plus)
    cluster_vars = cluster_sigmas**2
    
    # Total covariance matrix
    Cov = cov_sn.copy()
    np.fill_diagonal(Cov, Cov.diagonal() + cluster_vars)
    
    try:
        L = scipy.linalg.cho_factor(Cov, lower=True)
        #log_det = 2 * np.sum(np.log(np.diag(L[0])))
        log_det = 2 * np.sum(np.log(np.sqrt(2 * np.pi) * np.diag(L[0])))
        x = scipy.linalg.cho_solve(L, diff)
        chi2 = np.dot(diff, x)
        return -0.5 * (chi2 + log_det)
    except scipy.linalg.LinAlgError:
        return -np.inf

def ln_prior(theta):
    M, dm, eta = theta
    if not (-20.5 < M < -18.0): return -np.inf
    if not (-1.0 < dm < 1.0): return -np.inf
    if not (-1.0 < eta < 1.0): return -np.inf
    return 0.0

def ln_prob(theta, z, da, sp, sm, mb, cov):
    lp = ln_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + ln_likelihood(theta, z, da, sp, sm, mb, cov)

# =============================================================================
# 4. Main program
# =============================================================================
if __name__ == "__main__":
    
    # 1. Initialization
    z, dA, sp, sm, mb, cov, N = load_data()
    
    print(f"\n🚀 Starting MCMC analysis for (M_0, epsilon, eta)...")
    initial = np.array([-19.25, 0.0, 0.0])
    pos = [initial + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
    
    # 2. Run MCMC
    start_time = time.time()
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                        args=(z, dA, sp, sm, mb, cov),
                                        pool=pool if use_multiprocessing else None)
        sampler.run_mcmc(pos, nsteps, progress=True)
        
    print(f"\n✅ MCMC completed; elapsed time: {time.time() - start_time:.2f} s")
    
    # =========================================================================
    # 3. Post-process the chain with cosmo_tools
    # =========================================================================
    
    # Extract the flattened posterior chain
    flat_samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)
    labels = [r'M_0', r'\varepsilon', r'\eta'] # LaTeX labels used in the plots
    
    # A. Compute marginalized statistics
    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    
    # B. Evaluate the likelihood used for AIC/BIC
    # Use posterior medians as the representative parameter vector
    best_params = [s['median'] for s in stats]
    lnL_best = ln_likelihood(best_params, z, dA, sp, sm, mb, cov)
    
    # C. Print the parameter constraints
    cosmo_tools.print_results(stats, lnL_best=lnL_best, num_data=N)    
    # 2. Publication-quality GetDist plot with shaded credible intervals
    cosmo_tools.plot_getdist_advanced(flat_samples, labels, stats, filename="Cluster_Result_GetDist1.pdf")
