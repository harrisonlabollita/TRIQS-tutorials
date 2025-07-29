"""
03_crpa.py
==========

Script for a minimal constrained RPA (cRPA) calculation with CoQuí.

This script underlies the tutorial notebook `03_crpa_screened_interactions.ipynb`,
which breaks it into stages and explains each concept with hands-on exercises.

Notes:
------
- Parameters are chosen to run quickly for tutorial purposes.  
"""

from mpi4py import MPI
import coqui
from coqui.utils.imag_axes_ft import IAFT

# Create CoQui MPI handler and set logging verbosity
coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# --- Phase 1: Problem setup ---
# Mean-field (DFT) description of the target system
mf_params = {
    "prefix": "svo",
    "outdir": "data/qe_inputs/svo/222/out",
    "nbnd": 40
}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type="qe")

# Construct Coulomb Hamiltonian (in a compressed THC format)
thc_params = {
    "ecut": 40,
    "thresh": 1e-3
}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# --- Phase 2: Simulation (cRPA) ---
crpa_params = {
    "screen_type": "crpa",
    "input_type": "mf",
    "prefix": "crpa",
    "wannier_file": "data/qe_inputs/svo/222/mlwf/svo.mlwf.h5",
    "beta": 300,
    "wmax": 3.0,
    "iaft_prec": "medium"
}
Vloc, Uloc_t = coqui.downfold_local_coulomb(h_int=thc, params=crpa_params)

###### 
ft_kernel = IAFT(300, 3.0, "medium")
Uloc_iw = ir_kernel.tau_to_w_phsym(Uloc_t, stats='b')
