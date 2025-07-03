#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="interval-matching_bootstrap/"
subbase_dir="${base_dir}/bootstrap_benchmarks"
tagpath="${base_dir}/taglist.txt"

### input datasets ###
distlists_fpath="${subbase_dir}/real_distlists.csv"

### run parameters ###
samps=3
match_homdim=1
collate_type="dict"

### argument parsing ###
while getopts ":b:s:f:n:D:c:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) subbase_dir=${OPTARG}
    ;;
    f) distlists_fpath=${OPTARG}
    ;;
    n) samps=${OPTARG}
    ;;
    D) match_homdim=${OPTARG}
    ;;
    c) collate_type=${OPTARG}
    ;;
    \?) echo "Invalid option -${OPTARG}" >&2
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
prev_scr="${base_dir}/utils_match/prevalence.py"

distlists=$(cat ${distlists_fpath})

for distlist in ${distlists}
do
	echo ""
	echo "Pulling from ${distlist}"
    	for distname in $(cat $distlist)
    	do
		data_label=$(basename ${distname} | cut -d. -f 1)
		echo "Computing prevalence scores for phom bars corresponding to ${data_label}"

		# Name the submission script for this dataset type
		sbatch_fpath=$(dirname ${distname})"/do_phoms_${data_label}"

		# Create bookkeeping directories
		outdir=$(dirname ${distname})"/phom_data_${data_label}"
		matchdir="${outdir}/matching"

		verbose_match_fpath="${matchdir}/verbose_match_dim${match_homdim}_n${samps}_${collate_type}.txt"
		prevscore_fpath="${outdir}/prevalence_scores_dim${match_homdim}_n${samps}_${collate_type}.txt"
		B1match_fpath="${outdir}/B1match_dim${match_homdim}_n${samps}.txt"

		# Computes and saves prevalence scores from verbose matches
		# (add -v option to print prevalence scores to stdout)
		python ${prev_scr} -i ${verbose_match_fpath} -p ${prevscore_fpath} -m ${B1match_fpath}
	done
done
