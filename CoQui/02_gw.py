"""
02_gw.py
========

Script for a minimal GW calculation with CoQuí.

This script underlies the tutorial notebook `02_gw_electronic_structure.ipynb`, 
which breaks it into stages and explains each concept with hands-on exercises.

Notes:
------
- Parameters are chosen to run quickly for tutorial purposes.  
"""

from mpi4py import MPI
import coqui

# Create CoQui MPI handler and set logging verbosity 
coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# --- Phase 1: Problem setup ---
# Mean-field (DFT) description of the target system
mf_params = {
    "prefix": "si", 
    "outdir": "data/qe_inputs/si/out", 
    "nbnd": 10
}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type="qe")

# Construct Coulomb Hamiltonian (in a compressed THC format)
thc_params = {
    "ecut": 35,
    "thresh": 1e-2
}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# --- Phase 2: Simulation (GW) ---
gw_params = {
    "restart": True,
    "output": "gw",
    "beta": 300,
    "wmax": 3.0,
    "iaft_prec": "high",
    "niter": 1,
}
coqui.run_gw(params=gw_params, h_int=thc)
