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
    
    data = canvas.imshow(A_kw.T.real, origin='lower', 
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
        A_kw[ik, :] = -(1/np.pi)*np.linalg.inv((omegas[:, None, None] + mu + broadening - ek[ik,None, None] 
                         - Sigma_w.data[:])).trace(axis1=1,axis2=2).imag
    return A_kw


from triqs.gf import MeshImFreq, Gf, BlockGf, Block2Gf, make_gf_from_fourier, fit_hermitian_tail, make_hermitian
from triqs.gf.tools import make_zero_tail
from triqs.operators import c_dag, c
from triqs_cthyb import Solver

class SolverResults(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def __getattr__(self, key): 
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"SolverResults has no property {key}")
    def __repr__(self):
        return "\n".join([f"{key:<12}" for key in self.keys() ])
        
    __str__ = __repr__

def matrix_to_many_body_operator(h_loc_matrix, gf_struct):
    h_loc0_mat = {block : h_loc_matrix[ibl].real for ibl, (block, _) in enumerate(gf_struct) }
    c_dag_vec  = {block : np.matrix([[c_dag(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    c_vec      = {block : np.matrix([[c(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    return sum(c_dag_vec[block]*h_loc0_mat[block]*c_vec[block] for block, bl_size in gf_struct)[0,0]

def solve(Delta_iw, h_loc0, h_int, **solver_params):
    
    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    h_loc0 = matrix_to_many_body_operator(h_loc0, gf_struct)
    
    beta  = Delta_iw.mesh.beta
    n_iw = len(Delta_iw.mesh) // 2
    n_tau = 10*n_iw

    solver_params['measure_density_matrix'] = True
    solver_params['use_norm_as_weight']     = True

    S = Solver(gf_struct=gf_struct, beta=beta, n_iw=n_iw, n_tau=n_tau, delta_interface=True)

    for block, delta in S.Delta_tau:
        S.Delta_tau[block] << make_gf_from_fourier(
            Delta_iw[block],                                          # Δ(iω)
            S.Delta_tau[block].mesh,                                  # time mesh
            fit_hermitian_tail(Delta_iw[block], 
                               make_zero_tail(Delta_iw[block], 1))[0] # tail
            )
        
    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_params)
    Sigma_dynamic = S.Sigma_iw.copy()
    for bl, g in Sigma_dynamic: Sigma_dynamic[bl] << g - S.Sigma_Hartree[bl]
    
    return SolverResults(#Delta_tau   = S.Delta_tau, 
                         G_iw = S.G_iw,
                         #G_iw_raw = S.G_iw_raw,                      
                         #G_moments = S.G_moments,
                         G_tau = S.G_tau, 
                         #G_tau_accum = S.G_tau_accum, 
                         #G_l = S.G_l,                  
                         Sigma_Hartree = list(S.Sigma_Hartree.values()), 
                         Sigma_iw = S.Sigma_iw,
                         Sigma_dynamic = Sigma_dynamic,
                         #Sigma_iw_raw = S.Sigma_iw_raw, 
                         Sigma_moments = S.Sigma_moments,
                         #O_tau = S.O_tau,    
                         #G2_iw = S.G2_iw, 
                         #G2_iw_nfft = S.G2_iw_nfft, 
                         #G2_iw_ph = S.G2_iw_ph, 
                         #G2_iw_ph_nfft = S.G2_iw_ph_nfft, 
                         #G2_iw_pp = S.G2_iw_pp,
                         #G2_iw_pp_nfft = S.G2_iw_pp_nfft, 
                         #G2_iwll_ph = S.G2_iwll_ph, 
                         #G2_iwll_pp = S.G2_iwll_pp,
                         #G2_tau = S.G2_tau, 
                         #asymmetry_G_tau = S.asymmetry_G_tau,
                         auto_corr_time =  S.auto_corr_time, 
                         average_order = S.average_order, 
                         average_sign = S.average_sign, 
                         density_matrix = S.density_matrix,
                         h_loc_diagonalization = S.h_loc_diagonalization, 
                         orbital_occupations = S.orbital_occupations, 
                         performance_analysis = S.performance_analysis, 
                         perturbation_order = S.perturbation_order, 
                         perturbation_order_total = S.perturbation_order_total,
                        )
