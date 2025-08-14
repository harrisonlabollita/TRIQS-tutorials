import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_band_structure(kpts, bands,
                        fermi_level=0.0, 
                        high_symm_points= None,
                        high_symm_labels= None,
                        axes = None
                         ):
    axes = plt.gca() if not axes else axes
    
    for b in range(len(bands)): axes.plot(kpts, bands[b, :]-fermi_level, 'k-', lw=1)
    if high_symm_points:
        for k in high_symm_points: axes.axvline(k, color='k', ls='dotted', lw=0.5)
    axes.axhline(0.0 if fermi_level != 0 else fermi_level, color='k', ls='dotted', lw=0.5)
    axes.set_xticks(high_symm_points); axes.set_xlim(min(high_symm_points), max(high_symm_points));
    if high_symm_labels:
        assert len(high_symm_labels) == len(high_symm_points)
        axes.set_xticklabels(high_symm_labels)
    axes.tick_params(axis='x', which='both', length=0)

def plot_wannier_bands(kpts, bands, axes=None, **kwargs):
    axes = plt.gca() if not axes else axes
    for b in range(len(bands)): axes.plot(kpts, bands[b, :], '-', **kwargs)

def plot_spectral_function(Akw, k_lin, w_min, w_max, k_ticks, k_labels, axes=None, **kwargs):
    axes = plt.gca() if not axes else axes
    
    data = axes.imshow(Akw.T.real, origin='lower', 
                      aspect='auto', 
                      extent=(min(k_lin), max(k_lin), w_min, w_max),
                      **kwargs
                     )
    axes.set_xlim(min(k_lin), max(k_lin))
    axes.set_xticks(k_ticks)
    axes.set_xticklabels(k_labels)
    axes.set_ylabel(r'$\omega$')
    axes.tick_params(axis='x', which='both', length=0)
    axes.tick_params(axis='y', which='both', direction='out')

def momentum_resolved_spectral_function(e_k, mu, Sigma_w, broadening = 0.1j):
    n_kpts  = len(e_k)
    n_ws    = len(Sigma_w.mesh)
    A_kw  = np.zeros((n_kpts, n_ws))
    omegas = np.fromiter(Sigma_w.mesh, float)
    Id = np.eye(e_k.shape[-1])
    for ik in range(n_kpts):
        A_kw[ik, :] = -(1/np.pi)*np.linalg.inv(( omegas[:,None,None] + (mu + broadening)*Id - e_k[ik] - Sigma_w.data[:])
                                          ).trace(axis1=1,axis2=2).imag
    return A_kw
