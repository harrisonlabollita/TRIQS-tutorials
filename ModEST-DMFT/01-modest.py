import numpy as np 
from utils.solvers import solve
import triqs_modest as modest
from triqs.gf import MeshImFreq, MeshDLRImFreq, BlockGf
from triqs.operators import n
from triqs.plot.mpl_interface import oplot


beta = 10.0     # inverse temperature
n_iw = 251      # number of Matsubara points
n_tau = 10*n_iw
mesh = MeshImFreq(beta, S='Fermion', n_iw=251) # Matsubara mesh

U = 3.6
h_int = U*n('up',0)*n('down',0)
solver_params = dict(n_iw=251, n_tau=2510, length_cycle=80, 
                     n_cycles = int(5e+5), n_warmup_cycles = int(1e+3), 
                     perform_tail_fit=True, fit_min_w=6, fit_max_w=10, 
                     imag_threshold = 1e-6
                     )

seedname = 'data/mlwf/lco'
target_density = 1.0
spin_kind = "NonPolarized"

shells = [modest.AtomicShellT(l=2,dim=1,dft_idx=0,cls_idx=0)]

obe = modest.one_body_elements_from_wannier90(seedname, spin_kind, shells)

mu = modest.find_chemical_potential(target_density, obe, mesh, modest.BzIntOptions(), verbosity=True) 

hloc0_C =  modest.impurity_levels(obe) - mu
hloc0   = np.asarray(hloc0_C[0],dtype=complex).reshape(2,1,1) # Block2Matrix -> BlockMatrix

Gloc_C  = modest.gloc(mesh, obe, mu, opt)

Gloc = BlockGf(gf_struct=[('up',1), ('down',1)], mesh=mesh) # Block2Gf -> BlockGf
Gloc['up']   << Gloc_C['0','up']
Gloc['down'] << Gloc_C['0','down']

Delta_iw = modest.hybridization(hloc0, Gloc)

# solve!
solver_results = solve(Delta_iw, hloc0, h_int, **solver_params)

# update Sigma!
Sigma_static_C  = np.asarray(solver_results.Sigma_Hartree, dtype=complex).reshape(1,2,1,1) # BlockMatrix -> Block2Matrix

Sigma_dynamic_C  = Gloc_C.copy() # BlockGf -> Block2Gf
Sigma_dynamic_C['0','up']  << solver_results.Sigma_dynamic['up']
Sigma_dynamic_C['0','down'] << solver_results.Sigma_dynamic['down']

# update mu!
mu     = modest.find_chemical_potential(target_density, obe, Sigma_dynamic_C, Sigma_static_C, opt, verbosity=True)

# update Gloc!
Gloc_C  = modest.gloc(mesh, obe, mu, opt)
Gloc['up']   << Gloc_C['0','up']
Gloc['down'] << Gloc_C['0','down']

# update hloc0! εd - μ
hloc0_C = modest.impurity_levels(obe) - mu
hloc0   = np.asarray(hloc0_C[0],dtype=complex).reshape(2,1,1) # Block2Matrix -> BlockMatrix

# update Δ!
Delta_iw = modest.hybridization(hloc0, Gloc, solver_results.Sigma_dynamic, solver_results.Sigma_Hartree)

print(f"Δn = |n_lattice - n_impurity| = {abs(Gloc.total_density()-solver_results.G_iw.total_density())}")
