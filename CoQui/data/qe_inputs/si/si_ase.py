#!/usr/bin/python
# pyright: reportUnusedExpression=false

import os
import numpy as np
np.set_printoptions(legacy='1.25')

from ase import io
from ase.dft.kpoints import bandpath

from py_w90_driver.kpts_utils import gen_kpts_list
from py_w90_driver.qe import run_pwscf, run_pw2coqui, setup_nscf_nosym, copy_scf, read_kpoints
from py_w90_driver.wan90 import run_w90, wan2h5
from py_w90_driver.coqui import run_unfold_wfc


"""Common Parameters"""
mpi_exe = "mpirun"       # mpi executable
n_cores = None           # customized number of processors for mpi, otherwise all processors will be used.
mpi_nk = 1               # number of k-pools in parallelization in QE
qe_bin_dir = None
coqui_bin_dir = None
run_wan = True

seedname = 'si'
outdir = './out'
basename = os.path.basename(outdir.rstrip("/"))
parent_dir = os.path.dirname(outdir.rstrip("/"))
outdir = os.path.join(parent_dir, basename)

""" Atomic Structure read from cif """
atoms = io.read('../../../crys_struct/Si.cif', format='cif')
special_points = {
        'G': [0.00, 0.00, 0.00],
        'W': [0.50, 0.25, 0.75],
        'X': [0.50, 0.00, 0.50],
        'L': [0.50, 0.50, 0.50]
}
symm_kpath = bandpath(path=['W', 'G', 'G', 'X', 'X', 'W', 'W', 'L', 'L', 'G'], special_points=special_points, cell=atoms.cell)
mp_scf = (8, 8, 8)
mp_nscf = (2, 2, 2)
nbnd_coqui = 80
nbnd_wan = 20

""" Quantum Espresso Processes """

pseudopotentials = {'Si': 'Si_ONCV_PBE-1.2.upf'}
path_to_pseudopotentials = '../../../pseudo'

# parameters written to espresso.pwi
input_data = {
    'outdir': outdir,
    'prefix': seedname,
    'verbosity': 'high',
    'system': {
        'ibrav': 0,
        'nat': 5,
        'ntyp': 3,
        'ecutwfc': 50.0,
        'occupations': 'smearing',
        'smearing': 'm-p',
        'degauss': 0.0015,
        'input_dft':  'pbe'
    },
    'electrons': {'conv_thr': 1e-10, 'mixing_beta': 0.7, 'diagonalization': 'david'},
}

# scf
qe_params = {'atoms': atoms,
             'pseudopotentials': pseudopotentials,
             'path_pseudo': path_to_pseudopotentials,
             'input_data': input_data,
             'mp_grid': mp_scf}
run_pwscf(ase_atoms=qe_params['atoms'], input_data=qe_params['input_data'],
          pseudo_dir=qe_params['path_pseudo'], pseudopotentials=qe_params['pseudopotentials'],
          kpts=qe_params['mp_grid'], mpi_exe=mpi_exe, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)

# nscf w/ symmetries
input_data.update({'calculation': 'nscf'})
input_data['system']['nbnd'] = nbnd_coqui
input_data['system']['force_symmorphic'] = True
qe_params['mp_grid'] = mp_nscf
run_pwscf(ase_atoms=qe_params['atoms'], input_data=qe_params['input_data'],
          pseudo_dir=qe_params['path_pseudo'], pseudopotentials=qe_params['pseudopotentials'],
          kpts=qe_params['mp_grid'], mpi_exe=mpi_exe, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)

# pw2coqui
run_pw2coqui(qe_params['input_data']['prefix'], qe_params['input_data']['outdir'],
                       mpi_exe=mpi_exe, n_cores=1, bin_dir=qe_bin_dir)

