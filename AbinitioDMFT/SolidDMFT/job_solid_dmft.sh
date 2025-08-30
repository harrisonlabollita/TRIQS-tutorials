#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=lco.dmft.out
#SBATCH --error=lco.dmft.err

module load triqs/3.3

mpirun solid_dmft
