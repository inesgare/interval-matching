# SLURM bash scripting 

Bash shell code to distribute topological bootstrapping workflow to a job scheduler. Assumes a SLURM interface and a problem scale/complexity that warrants nested automation of I/O organization and job submission. Script details are as follows:

- `bootstrap_distances.sh`: distributes computation of distances between diagrams with matched cycles
- `calc_dists.sh`: distributes computation of metric (Gram) matrices over dataset(s)
- `calc_dists_w_perms.sh`: distributes computation of metric (Gram) matrices over permuted dataset(s)
- `collate_matches.sh`: distributes aggregation of cycle-matched data into compact dictionary structure
- `match_bootstraps.sh`: distributes cycle-matching computation
- `phom_bootstraps.sh`: distributes calls to Ripser/Ripser-image
- `prevwt_PDs.sh`: distributes visualization for prevalence-weighted persistence diagrams
- `permtest_distances.sh`: distributes computation of diagram distances between permuted-null persistence diagrams
- `prevalence.sh`: distributes computation of prevalence (i.e., cycle-match quality) information
- `submit_bsdist_sbatch.sh`: SLURM-submits computation of distance between diagrams of matched cycles
- `submit_distmtx_sbatch.sh`: SLURM-submits computation of metric (Gram) matrix
- `submit_permdist_sbatch.sh`: SLURM-submits computation of distances between diagrams and their null-permutation generated counterparts
- `submit_match_sbatch.sh`: SLURM-submits computation of matched cycles
- `submit_ripser_sbatch.sh`: SLURM-submits Ripser (or Ripser-image) call
