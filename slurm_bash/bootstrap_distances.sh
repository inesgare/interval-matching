#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="interval-matching_bootstrap"
parent_outdir="${base_dir}/bar_analysis/bootstrap_distances/PROFUMO_vs_null"
tagfile="${base_dir}/subsampling/taglist100k_90p_famstruct.txt"

### input datasets ###
Xdistlists_fpath="${parent_outdir}/distlistX_set.csv"
Ydistlists_fpath="${parent_outdir}/distlistY_set.csv"

### run parameters ###
count=100
bsdist_homdim=1
use_affinity=true
verbose=true

mem_gb=10

### argument parsing ###
while getopts ":b:o:X:Y:t:n:D:a:v:m:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    o) parent_outdir=${OPTARG}
    ;;
    X) Xdistlists_fpath=${OPTARG}
    ;;
    Y) Ydistlists_fpath=${OPTARG}
    ;;
    t) tagfile=${OPTARG}
    ;;
    n) count=${OPTARG}
    ;;
    D) bsdist_homdim=${OPTARG}
    ;;
    a) use_affinity=${OPTARG}
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
bsdist_scripter="${base_dir}/src_bash/submit_bsdist_sbatch.sh"

### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

Xdistlists=$(cat ${Xdistlists_fpath})
Ydistlists=$(cat ${Ydistlists_fpath})

if [ -s $tagfile ]
then
	tags=$( cat ${tagfile} | head -$count )
else
	echo "Error: no tagfile given"
	exit
fi

echo "Computing bootstrapped distances between spaces X and Y pulled from:"
echo "X: ${Xdistlists}"
echo "Y: ${Ydistlists}"

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

		for Ydistlist in ${Ydistlists}
		do
			for Ydistname in $(cat $Ydistlist)
			do
				Ydata_label=$(basename ${Ydistname} | cut -d. -f 1)
				Y_indir=$(dirname ${Ydistname})"/phom_data_${Ydata_label}"
				
				barY_fpath="${Y_indir}/bars_X.txt"

				printf "barY_fpath: \n$(ls $barY_fpath)\n"
				
				# script submitter for bsdisting job
				${bsdist_scripter} -x ${barX_fpath} -y ${barY_fpath} -D ${bsdist_homdim} -T ${tagfile} -n ${count} -a ${use_affinity} -v ${verbose} -o ${X_outdir}
			done
		done
	done
done
