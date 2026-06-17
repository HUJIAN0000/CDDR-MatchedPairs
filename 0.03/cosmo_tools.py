# -*- coding: utf-8 -*-
'''
cosmo_tools.py

Purpose
-------
Post-processing utilities for the MCMC analyses in the associated CDDR study. The module computes marginalized credible intervals and information criteria, and generates corner and GetDist triangle plots suitable for publication.

Associated manuscript: 'A model-independent test of the cosmic distance-duality relation using galaxy clusters and Type Ia supernovae matched pairs'.

@author:Jian Hu
Email:dg1626002@smail.nju.edu.cn
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

# Import optional plotting libraries
try:
    import corner
    from getdist import plots, MCSamples
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️ cosmo_tools: neither 'corner' nor 'getdist' was detected; plotting functions are unavailable.")

# ==========================================
# 1. Statistical summaries
# ==========================================
def calculate_stats(samples, labels):
    """
    Compute summary statistics for an MCMC chain.
    Compute the [2.275, 16, 50, 84, 97.725] percentiles corresponding to the 1-sigma and 2-sigma intervals.
    Return a list of dictionaries containing the summary statistics.
    """
    ndim = samples.shape[1]
    stats_list = []
    
    for i in range(ndim):
        # Compute the relevant posterior quantiles:
        # 2.275% (2-sigma lower), 16% (1-sigma lower), 50% (median), 84% (1-sigma upper), and 97.725% (2-sigma upper)
        p_vals = np.percentile(samples[:, i], [2.275, 16, 50, 84, 97.725])
        p2_low, p1_low, median, p1_high, p2_high = p_vals
        
        # Upper and lower 1-sigma uncertainties
        q_up = p1_high - median
        q_down = median - p1_low
        
        stat = {
            'name': labels[i], # Parameter label
            'median': median,
            'upper': q_up,
            'lower': q_down,
            'best': median,
            '1sigma': [p1_low, p1_high],
            '2sigma': [p2_low, p2_high], # Store the 2-sigma interval for plotting
            # Generate the LaTeX-formatted plot title
            'title_fmt': f"{labels[i]} = {median:.3f}_{{-{q_down:.3f}}}^{{+{q_up:.3f}}}"
        }
        stats_list.append(stat)
        
    return stats_list

def print_results(stats_list, lnL_best=None, num_data=None):
    """
    Print a standardized results table and compute AIC/BIC when lnL and the number of data points are supplied.
    """
    ndim = len(stats_list)
    
    print("\n" + "="*50)
    print("MCMC constraints (median with upper and lower 1-sigma uncertainties)")
    print("="*50)
    
    for st in stats_list:
        print(f"{st['name']:<10} = {st['median']:.4f}  +{st['upper']:.4f}  -{st['lower']:.4f}")
        
    if lnL_best is not None:
        print("-" * 50)
        print(f"Best lnL = {lnL_best:.2f}")
        
        if num_data is not None:
            # AIC = 2k - 2lnL
            AIC = 2 * ndim - 2 * lnL_best
            # BIC = k ln(N) - 2lnL
            BIC = ndim * np.log(num_data) - 2 * lnL_best
            print(f"AIC      = {AIC:.2f}")
            print(f"BIC      = {BIC:.2f}")
    print("="*50 + "\n")

# ==========================================
# 2. Plotting utilities
# ==========================================
def plot_corner(samples, labels, filename="corner.png", truths=None):
    """
    Generate a standard corner plot.
    """
    if not HAS_PLOT: return
    print(f"📊 Generating corner plot: {filename} ...")
    
    fig = corner.corner(
        samples, 
        bins=30, 
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        smooth=True, 
        smooth1d=True,
        plot_contours=True, 
        fill_contours=True,
        color="darkgreen", # Retain the plotting style used by the analysis scripts
        truths=truths,
        title_fmt=".3f"
    )
    fig.savefig(filename)
    print(f"✅ Corner plot saved")

def plot_getdist_advanced(samples, labels, stats_list, filename="getdist_result.pdf"):
    """
    Generate a publication-quality GetDist triangle plot following the style of CDDRSNGCV1.py.
    1. Use blue (#1f77b4) for the two-dimensional contours.
    2. Apply two-level shading to the one-dimensional distributions:
       - tan for the 2-sigma interval (alpha=0.4)
       - sienna for the 1-sigma interval (alpha=0.7)
    3. Add a legend.
    """
    if not HAS_PLOT: return
    print(f"🎨 Generating publication-quality GetDist plot: {filename} ...")
    
    # Simple parameter names for internal indexing
    names = [f"p{i}" for i in range(len(labels))]
    
    # Create the MCSamples object and configure smoothing
    mc_samples = MCSamples(samples=samples, names=names, labels=labels,
                           settings={'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.7})

    # Set the figure width
    g = plots.get_subplot_plotter(width_inch=8)
    
    # Draw the two-dimensional credible contours in blue
    g.triangle_plot(mc_samples, filled=True, contour_colors=['#1f77b4'])

    ndim = len(labels)
    has_leg = False # Track whether a legend label has already been added
    
    # Apply the custom two-level shading to the diagonal panels
    for i in range(ndim):
        ax = g.subplots[i, i]
        st = stats_list[i] # Retrieve the corresponding summary statistics
        
        # Set the title with asymmetric uncertainties
        ax.set_title(f"${st['title_fmt']}$", fontsize=12)
        
        # Retrieve the one-dimensional density estimate
        dens = mc_samples.get1DDensity(names[i])
        
        if dens:
            # --- Shade the 2-sigma region (tan) ---
            m2 = (dens.x >= st['2sigma'][0]) & (dens.x <= st['2sigma'][1])
            ax.fill_between(dens.x[m2], 0, dens.P[m2], 
                            color='tan', alpha=0.4, 
                            label=r'$2\sigma$ Region' if not has_leg else None)
            
            # --- Shade the 1-sigma region (sienna) ---
            m1 = (dens.x >= st['1sigma'][0]) & (dens.x <= st['1sigma'][1])
            ax.fill_between(dens.x[m1], 0, dens.P[m1], 
                            color='sienna', alpha=0.7, 
                            label=r'$1\sigma$ Region' if not has_leg else None)
            
            # Draw the median as a dashed line
            ax.axvline(st['median'], color='black', ls='--', lw=1.5)
            
            # Mark the legend entry as added to avoid duplicates
            has_leg = True

    # Add the global legend
    # Obtain handles and labels from the first panel
    handles, leg_labels = g.subplots[0, 0].get_legend_handles_labels()
    if handles:
        g.fig.legend(handles, leg_labels, loc='upper right', 
                     bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)

    # Save the figure
    g.export(filename)
    print(f"✅ GetDist plot saved")
