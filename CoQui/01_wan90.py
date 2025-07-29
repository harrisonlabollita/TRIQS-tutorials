from mpi4py import MPI
import coqui

# Create CoQui MPI handler and set logging verbosity 
coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# 1) Build a mean-field handle from existing QE outputs
params = {
  "prefix": "svo",                    # QE prefix (matches {prefix}.save)
  "outdir": "data/qe_inputs/svo/222/out", # QE outdir containing {prefix}.save/
  "nbnd": 40                          # number of bands read from QE outputs
}
mf = coqui.make_mf(coqui_mpi, params=params, mf_type="qe")

# 2) Run Wannier90 through CoQuí
w90_params = {
  "prefix": "svo",     # equivalent to wannier90's seedname 
}
coqui.wannier90(mf=mf, params=w90_params)
