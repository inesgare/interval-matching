#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="interval-matching_bootstrap"
parent_outdir="${base_dir}/bar_analysis/bootstrap_distances/PROFUMO_vs_null"

### input datasets ###
Xdistlists_fpath="${parent_outdir}/distlistX_set.csv"
Perm_distlists_fpath="${parent_outdir}/distlistPerm_set.csv"

### run parameters ###
permdist_homdim=1
verbose=true

mem_gb=10

### argument parsing ###
while getopts ":b:o:X:P:D:v:m:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    o) parent_outdir=${OPTARG}
    ;;
    X) Xdistlists_fpath=${OPTARG}
    ;;
    P) Perm_distlists_fpath=${OPTARG}
    ;;
    D) permdist_homdim=${OPTARG}
    ;;
    v) verbose=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

### paths to code ###
permdist_scripter="${base_dir}/src_bash/submit_permdist_sbatch.sh"

### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

Xdistlists=$(cat ${Xdistlists_fpath})
Perm_distlists=$(cat ${Perm_distlists_fpath})

echo "Computing bootstrapped distances between spaces X and Perms(X) pulled from:"
printf "X: \n${Xdistlists}\n"
printf "Perms: \n${Perm_distlists}\n"
echo ""

for Xdistlist in ${Xdistlists}
do
    	for Xdistname in $(cat $Xdistlist )
    	do
		Xdata_label=$(basename ${Xdistname} | cut -d. -f 1)
		X_indir=$(dirname ${Xdistname})"/phom_data_${Xdata_label}"
		
		barX_fpath="${X_indir}/bars_X.txt"
		printf "\nbarX_fpath: \n$(ls $barX_fpath)\n\n"

		X_outdir="${parent_outdir}/X_${Xdata_label}"
		mkdir -p "${X_outdir}/logs"

		for Perm_distlist in ${Perm_distlists}
		do
			${permdist_scripter} -x ${barX_fpath} -y ${Perm_distlist} -o ${X_outdir} -D ${permdist_homdim} -P 2 -Q 2 -v ${verbose} -m ${mem_gb}
		done
	done
done
