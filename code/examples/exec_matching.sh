#!/bin/bash
#SBATCH --job-name=matching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=30GB
#SBATCH --partition=cpu

# a 0-{N_resamp - 1}
#SBATCH -a 0-14

#SBATCH --output=out_files/exec_%A_%a.out

# COMMENTS:
# nodes = 1 means jobs on one machine : useful??
# %A = batch job number, %a = index or array of values
# ${SLURM_ARRAY_TASK_ID} = %a contain the same value
# change SBATCH -a [...-...] accordingly. Should be between 0 - N_resamp for cosmic web
# sbatch outputs written in out_files/ folder
# max simultaneous resources: ex: 8h, 1 cpu / task, 30 GB / cpu, 30 exps

#2,7,14,17,18,23,24,25,29

# launch Python script
python appli_matching.py ${SLURM_ARRAY_TASK_ID}

#PBS -o \$PBS_O_WORKDIR/out_files
#PBS -e \$PBS_O_WORKDIR/out_files
