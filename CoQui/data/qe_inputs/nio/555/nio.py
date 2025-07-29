#!/usr/bin/python
# pyright: reportUnusedExpression=false

import os
import numpy as np
np.set_printoptions(legacy='1.25')

from ase import io
from ase.dft.kpoints import bandpath
from ase.spacegroup import crystal

from py_w90_driver.kpts_utils import gen_kpts_list
from py_w90_driver.qe import run_pwscf, run_pw2coqui

"""Common Parameters"""
mpi_exe = "mpirun"
n_cores = None
mpi_nk = 4
qe_bin_dir = None
coqui_bin_dir = None

seedname = 'nio'
outdir = './out'
basename = os.path.basename(outdir.rstrip("/"))
parent_dir = os.path.dirname(outdir.rstrip("/"))
outdir = os.path.join(parent_dir, basename)

""" Atomic Structure read from cif """
a, b, c = 4.18633759, 4.18633759, 4.18633759
alpha, beta, gamma = 90, 90, 90
group = 225
atoms = crystal(symbols=['Ni', 'O'],
             basis=[(0.0, 0.0, 0.0),(0.00, 0.00, 0.50)],
             spacegroup=group,
             cellpar=[a, b, c, alpha, beta, gamma], primitive_cell=True)
special_points = {
        'K': [0.375, 0.375, 0.75],
        'W': [0.50, 0.25, 0.75],
        'L': [0.50, 0.50, 0.50],
        'X': [0.50, 0.00, 0.50],
        'G': [0.00, 0.00, 0.00],
}
symm_kpath = bandpath(path=['K', 'G', 'G', 'X', 'X', 'W', 'W', 'L', 'L', 'G'], special_points=special_points, cell=atoms.cell)
mp_scf = (10,10,10)
mp_nscf = (5,5,5)
nbnd_coqui = 80

""" Quantum Espresso Processes """

pseudopotentials = {'Ni': 'Ni_ONCV_PBE-1.2.upf',
                    'O': 'O_ONCV_PBE-1.2.upf'}
path_to_pseudopotentials = '../../../pseudo/'

# parameters written to espresso.pwi
input_data = {
    'outdir': outdir,
    'prefix': seedname,
    'verbosity': 'high',
    'system': {
        'ibrav': 0,
        'nat': 2,
        'ntyp': 2,
        'ecutwfc': 100.0,
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
          kpts=qe_params['mp_grid'], mpi_exe=mpi_exe, n_cores=n_cores, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)


# nscf w/ symmetries
input_data.update({'calculation': 'nscf'})
input_data['system']['nbnd'] = nbnd_coqui
input_data['system']['force_symmorphic'] = True
input_data['electrons']['diago_full_acc'] = True
qe_params['mp_grid'] = mp_nscf
conv_thr = np.array([1e-4, 1e-7, 1e-10])
for i, e in enumerate(conv_thr):
    if e != 1e-4:
        input_data['electrons']['startingwfc'] = 'file'
    input_data['electrons']['conv_thr'] = e
    run_pwscf(ase_atoms=qe_params['atoms'], input_data=qe_params['input_data'],
              pseudo_dir=qe_params['path_pseudo'], pseudopotentials=qe_params['pseudopotentials'],
              kpts=qe_params['mp_grid'], mpi_exe=mpi_exe, n_cores=n_cores, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)

# pw2aimbes
run_pw2coqui(qe_params['input_data']['prefix'], qe_params['input_data']['outdir'],
                       mpi_exe=mpi_exe, n_cores=1, bin_dir=qe_bin_dir)
