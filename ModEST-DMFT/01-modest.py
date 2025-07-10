import numpy as np
from common import *
import bzsummation as modest
from triqs.gf import MeshImFreq, MeshDLRImFreq
from triqs.operators import n
from triqs.plot.mpl_interface import oplot

hdf5_filename = 'data/mlwf/lco_wannier.h5'
target_density, obe = modest.one_body_elements_from_dft_converter(hdf5_filename)
E = modest.make_default_embedding(obe.C_space)

#checkpoint = modest.DMFTCheckpoint('la2cuo4', modest.InitialData(obe=obe, embed=E))

beta = 10.0 # inverse temperature
#mesh = MeshImFreq(beta, S='Fermion', n_iw=251) # Matsubara mesh
mesh = MeshDLRImFreq(beta, statistic='Fermion', w_max = 4, eps=1e-6) # Matsubara mesh

U = 3.6
h_int = U*n('up_0',0)*n('down_0',0)
solver_params = dict(length_cycle=60, n_cycles = int(5e+6), n_warmup_cycles = int(1e+4), perform_tail_fit=False) #fit_min_w=6,
                     #fit_max_w=10, imag_threshold = 1e-6)

mu = modest.find_chemical_potential(target_density, obe, beta, verbosity=False) # verbosity broken!

hloc_C =  modest.impurity_levels(obe) - mu

hloc0 = E.extract(hloc_C)[0]

Gloc = E.extract(modest.gloc(mesh, obe, mu))[0]

Delta_iw = modest.extract_delta(hloc0, Gloc)

<<<<<<< HEAD
U = 3.0
h_int = U*n('up_0',0)*n('down_0',0)

solver_params = dict(length_cycle=60, n_cycles = int(5e+5),
                          n_warmup_cycles = int(1e+4),
                          perform_tail_fit=True, fit_min_w=6, fit_max_w=10,
                          imag_threshold = 1e-6)

n_dmft_loops = 10

n_dmft_loops = 20
for n_iter in range(n_dmft_loops):
    print(f"DMFT iteration= {n_iter}")

    # solve!
    solver_results = solve(Delta_iw, hloc0, h_int, **solver_params)

    # update Sigma!
    Sigma_hartree_C  = E.embed( [solver_results.Sigma_Hartree] )
    Sigma_dynamic_C  = E.embed([ solver_results.Sigma_dynamic ])

    # update mu!
    mu     = modest.find_chemical_potential(target_density, obe, Sigma_dynamic_C, Sigma_hartree_C, verbosity=False)

    # update Gloc!
    Gloc   = E.extract(modest.gloc(obe, mu, Sigma_dynamic_C, Sigma_hartree_C))[0]

    # update hloc0! εd - μ
    hloc0 = E.extract(modest.impurity_levels(obe) - mu)[0]

    # update Δ!
    Delta_iw = modest.extract_delta(hloc0, Gloc, solver_results.Sigma_dynamic, solver_results.Sigma_Hartree)

    print(f"Δn = |n_lattice - n_impurity| = {abs(Gloc.total_density()-solver_results.G_iw.total_density())}")

    Sigma_dlr_full = make_gf_imfreq(make_gf_dlr(solver_results.Sigma_dynamic), 1025)

    plt.figure()
    oplot(Sigma_dlr_full['up_0'][0,0].imag, 'o-')
    plt.xlim(0, 20)
    plt.show()
