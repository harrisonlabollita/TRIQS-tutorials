from mpi4py import MPI
import coqui

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=2)

# construct MF from a dictionary 
mf_params = {
    "prefix": "si", 
    "outdir": "../out",
    "filetype": "h5",
    "nbnd": 20
}
si_mf = coqui.make_mf(mpi, mf_params, "qe")

# wannier90
w90_params = {
  "prefix": "si"
}
coqui.wannier90(si_mf, w90_params)
