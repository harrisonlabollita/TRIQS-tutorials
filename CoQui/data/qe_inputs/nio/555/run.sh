#!/bin/bash
#SBATCH -p ccq
#SBATCH -C genoa
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 1
#SBATCH -o nio_ase.o%j
#SBATCH -J nio_ase

module load llvm19-mkl-mod2.4 ab-initio-es/release

#OpenMP settings:
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

date
python nio.py 
date

