import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

def __plot_band_structure(canvas,
                          kpts, bands,
                          fermi_level=0.0,
                          high_symm_points= None,
                          high_symm_labels= None
                         ):

    for b in range(len(bands)): canvas.plot(kpts, bands[b, :]-fermi_level, 'k-', lw=1)
    if high_symm_points:
        for k in high_symm_points: canvas.axvline(k, color='k', ls='dotted', lw=0.5)
    canvas.axhline(0.0 if fermi_level != 0 else fermi_level, color='k', ls='dotted', lw=0.5)
    canvas.set_xticks(high_symm_points); canvas.set_xlim(min(high_symm_points), max(high_symm_points));
    if high_symm_labels:
        assert len(high_symm_labels) == len(high_symm_points)
        canvas.set_xticklabels(high_symm_labels)
    canvas.tick_params(axis='x', which='both', length=0)

def __plot_bands(canvas, kpts, bands, **kwargs):
    for b in range(len(bands)): canvas.plot(kpts, bands[b, :], '-', **kwargs)

def __plot_spectral_function(canvas, Akw, k_lin, k_ticks, k_labels, **kwargs):

    data = canvas.imshow(Akw.T.real, origin='lower',
                      aspect='auto',
                      extent=(min(k_lin), max(k_lin), -10,10),
                      **kwargs
                     )
    canvas.set_xlim(min(k_lin), max(k_lin))
    canvas.set_xticks(k_ticks)
    canvas.set_xticklabels(k_labels)
    canvas.set_ylabel(r'$\omega$')
    canvas.tick_params(axis='x', which='both', length=0)
    canvas.tick_params(axis='y', which='both', direction='out')


mpl.axes.Axes.plot_band_structure = lambda self, kpts, bands, **kwargs: __plot_band_structure(self, kpts, bands, **kwargs)
mpl.axes.Axes.plot_bands = lambda self, kpts, bands, **kwargs: __plot_bands(self, kpts, bands, **kwargs)
mpl.axes.Axes.plot_spectral_function = lambda self, Akw, k_lin, k_ticks, k_labels, **kwargs : __plot_spectral_function(self, Akw, k_lin, k_ticks, k_labels, **kwargs)


def momentum_resolved_spectral_function(e_k, mu, Sigma_w, broadening = 0.1j):
    n_kpts  = len(e_k)
    n_ws    = len(Sigma_w.mesh)
    A_kw  = np.zeros((n_kpts, n_ws))
    omegas = np.fromiter(Sigma_w.mesh, float)
    for ik in range(n_kpts):
        A_kw[ik, :] = -(1/np.pi)*np.linalg.inv((omegas[:, None, None] + mu + broadening - e_k[ik,None, None]
                         - Sigma_w.data[:])).trace(axis1=1,axis2=2).imag
    return A_kw
