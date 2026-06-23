# CDDR-MatchedPairs

**Model-independent tests of the cosmic distance-duality relation with galaxy clusters, Pantheon+ Type Ia supernovae, and DESI BAO**

This repository contains the data-analysis code associated with the manuscript:

> **A model-independent test of the cosmic distance-duality relation using galaxy clusters and Type Ia supernovae matched pairs**  
> Jian Hu, Yi Liu, Jian-Ping Hu, and Zhongmu Li

The analysis tests the cosmic distance-duality relation (CDDR) without assuming a background expansion model for the parameter inference. Galaxy-cluster angular-diameter distances are paired with Pantheon+ Type Ia supernovae at nearly identical distances, and supplementary analyses include four DESI BAO measurements.

## Scientific model

The generalized CDDR is parameterized as

$$
D_L(z)=D_A(z)(1+z)^2(1+\eta z),
$$

where the standard Etherington reciprocity relation corresponds to $\eta=0$.

A possible redshift evolution of the standardized Type Ia supernova absolute magnitude is modeled as

$$
M_B(z)=M_0+\epsilon z.
$$

The parameters $M_0$, $\epsilon$, and $\eta$ are sampled jointly. In analyses including DESI BAO, the sound horizon $r_d$ is either treated as a free parameter or constrained with the Gaussian Planck prior

$$
r_d=147.09\pm0.26\ \mathrm{Mpc}.
$$

> **Code notation:** the variable `dm` in the Python scripts denotes the supernova-evolution parameter $\epsilon$. It must not be confused with the transverse comoving distance $D_M$ used in the DESI BAO likelihood.

## Main features

- Model-independent galaxy-cluster–supernova matched-pair analysis.
- Baseline relative comoving-distance matching tolerance of 5%.
- Robustness test using a stricter 3% matching tolerance.
- Pantheon+ statistical and systematic sub-covariance matrices.
- Split-normal treatment of asymmetric galaxy-cluster distance uncertainties.
- Joint inference of $M_0$, $\epsilon$, and $\eta$.
- Supplementary DESI BAO analyses with free $r_d$ or a Planck prior.
- Standard, optimized CPU, and Numba-compiled implementations.
- Posterior summaries, AIC/BIC calculations, corner plots, and GetDist triangle plots.

## Repository structure

```text
CDDR-MatchedPairs/
├── README.md
├── 0.05/
│   ├── matched_result_cluster.csv
│   ├── cov_cluster_lens.txt
│   ├── cov_desi_sn.txt
│   ├── cosmo_tools.py
│   ├── test_cluster_M_dm_eta.py
│   ├── test_cluster_M_dm_eta_fast_cpu.py
│   ├── test_cluster_M_dm_eta_ultrafast_cpu.py
│   ├── test_cluster_M_dm_eta2.py
│   ├── test_cluster_M_dm_eta2_fast_cpu.py
│   ├── test_cluster_M_dm_eta2_ultrafast_cpu.py
│   ├── test_cluster_M_dm_eta2s.py
│   ├── test_cluster_M_dm_eta2s_fast_cpu.py
│   └── test_cluster_M_dm_eta2s_ultrafast_cpu.py
└── 0.03/
    ├── matched_result_cluster0.03.csv
    ├── cov_cluster_lens0.03.txt
    ├── cov_desi_sn.txt
    ├── cosmo_tools.py
    ├── test_cluster_M_dm_eta0.03.py
    ├── test_cluster_M_dm_eta0.03_fast_cpu.py
    ├── test_cluster_M_dm_eta20.03.py
    └── test_cluster_M_dm_eta20.03_fast_cpu.py
```

The directory names refer to the relative line-of-sight comoving-distance tolerance used to construct the matched pairs:

- `0.05/`: $\Delta D_C/D_C\leq5\%$; 38 matched galaxy-cluster–SNe Ia pairs.
- `0.03/`: $\Delta D_C/D_C\leq3\%$; 37 matched galaxy-cluster–SNe Ia pairs.

These values are distance-matching tolerances, not simple redshift cuts.

## Script guide

### Baseline 5% sample

| Script | Analysis |
|---|---|
| `test_cluster_M_dm_eta.py` | Cluster-only reference implementation; samples $M_0$, $\epsilon$, and $\eta$. |
| `test_cluster_M_dm_eta_fast_cpu.py` | Optimized CPU implementation of the cluster-only analysis. |
| `test_cluster_M_dm_eta_ultrafast_cpu.py` | Numba-compiled cluster-only implementation. |
| `test_cluster_M_dm_eta2s.py` | Cluster + DESI BAO analysis with free $r_d$. |
| `test_cluster_M_dm_eta2s_fast_cpu.py` | Optimized CPU version with free $r_d$. |
| `test_cluster_M_dm_eta2s_ultrafast_cpu.py` | Numba-compiled version with free $r_d$. |
| `test_cluster_M_dm_eta2.py` | Cluster + DESI BAO analysis with the Planck prior on $r_d$. |
| `test_cluster_M_dm_eta2_fast_cpu.py` | Optimized CPU version with the Planck prior. |
| `test_cluster_M_dm_eta2_ultrafast_cpu.py` | Numba-compiled version with the Planck prior. |

### Stricter 3% sample

| Script | Analysis |
|---|---|
| `test_cluster_M_dm_eta0.03.py` | Cluster-only 3% robustness analysis. |
| `test_cluster_M_dm_eta0.03_fast_cpu.py` | Optimized CPU version of the cluster-only 3% analysis. |
| `test_cluster_M_dm_eta20.03.py` | Cluster + DESI BAO 3% analysis with the Planck prior on $r_d$. |
| `test_cluster_M_dm_eta20.03_fast_cpu.py` | Optimized CPU version of the joint 3% analysis. |

The reference and accelerated scripts are designed to implement the same likelihood and priors. Their chains will not be bitwise identical because MCMC sampling is stochastic, but the inferred posterior constraints should agree within Monte Carlo uncertainty.

## Data files

- `matched_result_cluster.csv`: 38 matched galaxy-cluster–Pantheon+ pairs for the 5% sample.
- `matched_result_cluster0.03.csv`: 37 matched pairs for the 3% sample.
- `cov_cluster_lens.txt`: Pantheon+ sub-covariance matrix for the 5% matched supernova sample.
- `cov_cluster_lens0.03.txt`: Pantheon+ sub-covariance matrix for the 3% matched supernova sample.
- `cov_desi_sn.txt`: Pantheon+ covariance matrix for the four supernovae used with the DESI BAO anchors.

The galaxy-cluster distances are based on the X-ray/Sunyaev–Zel'dovich compilation of Bonamente et al. (2006). The supernova data are drawn from Pantheon+, and the supplementary BAO measurements are taken from the DESI BAO summary products used in the associated manuscript.

## Installation

Python 3.9 or later is recommended. Install the required packages with

```bash
python -m pip install numpy pandas scipy emcee matplotlib corner getdist threadpoolctl numba
```

Package roles:

- Core analyses: `numpy`, `pandas`, `scipy`, `emcee`.
- Plotting and posterior visualization: `matplotlib`, `corner`, `getdist`.
- Optimized CPU scripts: `threadpoolctl`.
- Ultra-fast scripts: `numba`.

`corner` and `getdist` are optional for parameter estimation itself, but they are required to generate all posterior figures.

## Running the analyses

The scripts use relative paths for their input files. Run each script from within its own data directory.

### 5% baseline cluster-only analysis

```bash
cd 0.05
python test_cluster_M_dm_eta.py
```

Recommended optimized version:

```bash
python test_cluster_M_dm_eta_fast_cpu.py
```

Numba-compiled version:

```bash
python test_cluster_M_dm_eta_ultrafast_cpu.py
```

### 5% cluster + DESI analysis with free $r_d$

```bash
cd 0.05
python test_cluster_M_dm_eta2s_fast_cpu.py
```

### 5% cluster + DESI analysis with the Planck prior

```bash
cd 0.05
python test_cluster_M_dm_eta2_fast_cpu.py
```

### 3% cluster-only robustness analysis

```bash
cd 0.03
python test_cluster_M_dm_eta0.03_fast_cpu.py
```

### 3% cluster + DESI + Planck robustness analysis

```bash
cd 0.03
python test_cluster_M_dm_eta20.03_fast_cpu.py
```

The scripts print marginalized posterior constraints and fit statistics to the terminal. They also generate corner plots in PNG format and GetDist triangle plots in PDF format in the current working directory.

## Reference results

The following constraints are reported in the associated manuscript and can be used as numerical checks. Uncertainties correspond to 68% credible intervals.

| Dataset | $M_0$ | $\epsilon$ | $\eta$ | $r_d$ (Mpc) |
|---|---:|---:|---:|---:|
| Clusters only, 5% | $-19.460^{+0.126}_{-0.124}$ | $-0.184^{+0.724}_{-0.574}$ | $0.050^{+0.348}_{-0.307}$ | — |
| Clusters only, 3% | $-19.411^{+0.134}_{-0.124}$ | $-0.154^{+0.721}_{-0.595}$ | $-0.044^{+0.343}_{-0.289}$ | — |
| Clusters + DESI, free $r_d$ | $-19.504^{+0.094}_{-0.101}$ | $-0.221^{+0.659}_{-0.515}$ | $0.123^{+0.333}_{-0.286}$ | $140.56^{+8.12}_{-8.11}$ |
| Clusters + DESI + Planck | $-19.481^{+0.086}_{-0.093}$ | $-0.239^{+0.645}_{-0.519}$ | $0.079^{+0.309}_{-0.267}$ | $147.078\pm0.259$ |
| Clusters (3%) + DESI + Planck | $-19.472^{+0.086}_{-0.093}$ | $-0.249^{+0.639}_{-0.506}$ | $0.069^{+0.292}_{-0.265}$ | $147.084\pm0.259$ |

Small differences between individual runs are expected because of finite-chain Monte Carlo noise, hardware differences, and random initialization.

## Reproducibility notes

1. Keep each covariance matrix in the same directory as its corresponding analysis script.
2. Do not interchange the 3% and 5% matched catalogs or covariance matrices.
3. The asymmetric cluster errors are selected dynamically according to the sign of the distance-modulus residual.
4. The DESI likelihood is evaluated in $D_M/r_d$ space and includes propagation of the matched-supernova covariance.
5. The accelerated scripts change the numerical implementation, not the intended statistical model.
6. Set an explicit random seed in the configuration section when deterministic chain initialization is required.

## Citation

Please cite the associated paper when using this code or the derived data products:

> Hu, J., Liu, Y., Hu, J.-P., & Li, Z. (2026). A model-independent test of the cosmic distance-duality relation using galaxy clusters and Type Ia supernovae matched pairs. *Astronomy & Astrophysics*, Forthcoming article. https://doi.org/10.1051/0004-6361/202659255

**BibTeX:**

```bibtex
@article{Hu2026CDDRMatchedPairs,
  author  = {Hu, Jian and Liu, Yi and Hu, Jian-Ping and Li, Zhongmu},
  title   = {A model-independent test of the cosmic distance-duality relation using galaxy clusters and Type Ia supernovae matched pairs},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659255},
  url     = {[https://doi.org/10.1051/0004-6361/202659255](https://doi.org/10.1051/0004-6361/202659255)},
  note    = {Forthcoming article}
}
```

The final journal DOI and Zenodo DOI should be added here after they are assigned.

## Author and contact

**Jian Hu**  
Email: [dg1626002@smail.nju.edu.cn](mailto:dg1626002@smail.nju.edu.cn)

## Acknowledgements

This repository makes use of the Pantheon+ supernova compilation, the Bonamente et al. galaxy-cluster sample, and DESI BAO measurements. Please also cite the original observational-data publications listed in the associated manuscript.
