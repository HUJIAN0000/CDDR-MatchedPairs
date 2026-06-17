# -*- coding: utf-8 -*-
'''
test_cluster_M_dm_eta20.03.py

Purpose
-------
Joint 3% matched-pair analysis combining 37 galaxy-cluster angular-diameter 
distances and their matched Pantheon+ SNe Ia with four DESI BAO D_M/r_d 
measurements. The parameters M_0, epsilon, eta, and r_d are sampled jointly, 
including the Gaussian Planck prior r_d = 147.09 +/- 0.26 Mpc.

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
import emcee
import time
from multiprocessing import Pool
import cosmo_tools

# =========================================================================
# 1. Data loader (all files are read once before MCMC sampling)
# =========================================================================
def load_all_data():
    print("Loading galaxy-cluster data...")
    df = pd.read_csv("matched_result_cluster0.03.csv")
    z_cl = df['z'].values
    dA_Mpc = df['DA(GPC)'].values * 1000.0
    sp_Mpc = df['SDA+'].values * 1000.0
    sm_Mpc = df['SDA-'].values * 1000.0
    
    COEFF = 5.0 / np.log(10.0)
    sp_mu = COEFF * (sp_Mpc / dA_Mpc)
    sm_mu = COEFF * (sm_Mpc / dA_Mpc)
    mb_cl = df['sn_matched_m_b_corr'].values
    cov_cl = np.loadtxt("cov_cluster_lens0.03.txt")
    
    print("Loading DESI BAO measurements and the corresponding Pantheon+ SNe Ia...")
    # DESI BAO measurements
    z_bao = np.array([0.510, 0.706, 0.930, 1.317])
    dm_over_rd_obs = np.array([13.62003, 16.84645, 21.70841, 27.78720])
    bao_var = np.array([0.06346622, 0.10197571, 0.07956752, 0.47656986])
    
    # Matched Pantheon+ SNe Ia (indices 1502, 1628, 1674, and 1687)
    mb_desi = np.array([22.8415, 23.6432, 24.2801, 25.5337])
    cov_desi_sn = np.loadtxt("cov_desi_sn.txt")
    
    # Return the cluster and DESI data blocks
    data_cl = (z_cl, dA_Mpc, sp_mu, sm_mu, mb_cl, cov_cl)
    data_desi = (z_bao, dm_over_rd_obs, bao_var, mb_desi, cov_desi_sn)
    return data_cl, data_desi

# =========================================================================
# 2. Likelihood and priors
# =========================================================================
def ln_likelihood(theta, data_cl, data_desi):
    M, dm, eta, rd = theta
    (z_cl, dA_Mpc, sp_mu, sm_mu, mb_cl, cov_cl) = data_cl
    (z_bao, dm_over_rd_obs, bao_var, mb_desi, cov_desi_sn) = data_desi
    
    # ------------------
    # A. Galaxy-cluster contribution
    # ------------------
    term_z_cl = (1 + z_cl)**2 * (1 + eta * z_cl)
    mu_cl_th = 5.0 * np.log10(dA_Mpc * term_z_cl) + 25.0
    mu_sn_cl = mb_cl - (M + dm * z_cl)
    diff_cl = mu_sn_cl - mu_cl_th
    
    sigmas_cl = np.where(diff_cl > 0, sm_mu, sp_mu)
    Cov_cl_tot = cov_cl.copy()
    np.fill_diagonal(Cov_cl_tot, Cov_cl_tot.diagonal() + sigmas_cl**2)
    
    try:
        L_cl = scipy.linalg.cho_factor(Cov_cl_tot, lower=True)
        log_det_cl = 2 * np.sum(np.log(np.sqrt(2 * np.pi) * np.diag(L_cl[0])))
        x_cl = scipy.linalg.cho_solve(L_cl, diff_cl)
        chi2_cl = np.dot(diff_cl, x_cl) + log_det_cl
    except scipy.linalg.LinAlgError:
        return -np.inf

    # ------------------
    # B. DESI BAO Chi2 
    # ------------------
    mu_sn_bao = mb_desi - (M + dm * z_bao)
    DL_sn = 10**((mu_sn_bao - 25.0) / 5.0)
    
    DM_th = DL_sn / ((1 + z_bao) * (1 + eta * z_bao))
    dm_over_rd_th = DM_th / rd
    diff_bao = dm_over_rd_obs - dm_over_rd_th
    
    J_diag = (np.log(10) / 5.0) * dm_over_rd_th
    J = np.diag(J_diag)
    Cov_sn_converted = J @ cov_desi_sn @ J.T
    
    Cov_bao_tot = np.diag(bao_var) + Cov_sn_converted
    
    try:
        L_bao = scipy.linalg.cho_factor(Cov_bao_tot, lower=True)
        log_det_bao = 2 * np.sum(np.log(np.sqrt(2 * np.pi) * np.diag(L_bao[0])))
        x_bao = scipy.linalg.cho_solve(L_bao, diff_bao)
        chi2_bao = np.dot(diff_bao, x_bao) + log_det_bao
    except scipy.linalg.LinAlgError:
        return -np.inf

    # ------------------
    # C. Planck sound-horizon prior
    # ------------------
    chi2_prior = ((rd - 147.09) / 0.26)**2

    return -0.5 * (chi2_cl + chi2_bao + chi2_prior)

def ln_prior(theta):
    M, dm, eta, rd = theta
    if not (-20.5 < M < -18.0): return -np.inf
    if not (-1.0 < dm < 1.0): return -np.inf
    if not (-1.0 < eta < 1.0): return -np.inf
    if not (130 < rd < 160): return -np.inf
    return 0.0

# Pass both data blocks explicitly to the posterior function
def ln_prob(theta, data_cl, data_desi):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, data_cl, data_desi)

# =========================================================================
# 3. Main program
# =========================================================================
if __name__ == "__main__":
    
    # 1. Load all input data once before sampling
    data_cl, data_desi = load_all_data()
    
    ndim = 4
    nwalkers = 32
    nsteps = 6000
    burn_in = 1500
    
    print("\n🚀 Starting the joint Pantheon+ MCMC analysis for (M_0, epsilon, eta, r_d)...")
    initial = np.array([-19.25, 0.0, 0.0, 147.1])
    pos = [initial + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
    
    start_time = time.time()
    
    use_multiprocessing = True
    
    if use_multiprocessing:
        with Pool() as pool:
            # Pass the cluster and DESI data blocks together as sampler arguments
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                            args=(data_cl, data_desi),
                                            pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                        args=(data_cl, data_desi))
        sampler.run_mcmc(pos, nsteps, progress=True)
        
    print(f"\n✅ MCMC completed; elapsed time: {time.time() - start_time:.2f} s")
    
    # Extract the posterior samples and generate plots
    flat_samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)
    labels = [r'M_0', r'\varepsilon', r'\eta', r'r_d']
    
    stats = cosmo_tools.calculate_stats(flat_samples, labels)
    best_params = [s['median'] for s in stats]
    lnL_best = ln_likelihood(best_params, data_cl, data_desi)
    
    cosmo_tools.print_results(stats, lnL_best=lnL_best, num_data=42)
    cosmo_tools.plot_getdist_advanced(flat_samples, labels, stats, filename="Pantheon_Joint_Result_GetDist0.03.pdf")
