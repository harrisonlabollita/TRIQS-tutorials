from mpi4py import MPI
import coqui

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# construct MF from a dictionary 
mf_params = {
    "prefix": "nio", 
    "outdir": "../out",
    "nbnd": 30
}
nio_mf = coqui.make_mf(mpi, mf_params, "qe")

# wannier90
w90_params = {
  "prefix": "nio"
}
coqui.wannier90(nio_mf, w90_params)
