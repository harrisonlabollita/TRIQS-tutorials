import numpy as np
from common import *
import bzsummation as modest
from triqs.gf import MeshImFreq, inverse
from triqs.operators import n

hdf5_filename = 'data/mlwf/lco_wannier.h5'
target_density, obe = modest.one_body_elements_from_dft_converter(hdf5_filename)
E = modest.make_default_embedding(obe.C_space)

beta = 10.0 # inverse temperature
mesh = MeshImFreq(beta, S='Fermion', n_iw=1025) # Matsubara mesh
mu = modest.find_chemical_potential(target_density, obe, beta, verbosity=False) # verbosity broken!
Gloc = E.extract(modest.gloc(mesh, obe, mu))[0]
hloc0 = E.extract(modest.impurity_levels(obe)-mu)[0]
Delta_iw = modest.extract_delta(hloc0, Gloc)

U = 3.0
h_int = U*n('up_0',0)*n('down_0',0)

solver_params = dict(length_cycle=60, n_cycles = int(5e+5),
                          n_warmup_cycles = int(1e+4),
                          perform_tail_fit=True, fit_min_w=6, fit_max_w=10,
                          imag_threshold = 1e-6)

n_dmft_loops = 10

for n_iter in range(n_dmft_loops):
    print(f"DMFT iteration= {n_iter}")
    # solve!
    solver_results = solve(Delta_iw, hloc0, h_int, **solver_params)
    # update Sigma!
    Sigma_C  = E.embed([ solver_results.Sigma_iw ])
    # update mu!
    mu     = modest.find_chemical_potential(target_density, obe, Sigma_C, verbosity=False)
    # update Gloc!
    Gloc   = E.extract(modest.gloc(obe, mu, Sigma_C, None))[0] # FIXME
    # update hloc0
    hloc0 = E.extract(modest.impurity_levels(obe)-mu)[0]
    # update Δ!
    Delta_iw = modest.extract_delta(hloc0, Gloc, solver_results.Sigma_iw)
    print(f"Δn = |n_lattice - n_impurity| = {abs(Gloc.total_density()-solver_results.G_iw.total_density())}")
