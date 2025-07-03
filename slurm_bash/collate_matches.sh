#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations"
subbase_dir="${base_dir}/bootstrap_benchmarks"
tagfile="${base_dir}/taglist.txt"

### input datasets ###
distlists_fpath="${subbase_dir}/real_distlists.csv"

### run parameters ###
samps=3
match_homdim=1

### argument parsing ###
while getopts ":b:s:f:t:D:n:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) subbase_dir=${OPTARG}
    ;;
    f) distlists_fpath=${OPTARG}
    ;;
    t) tagfile=${OPTARG}
    ;;
    D) match_homdim=${OPTARG}
    ;;
    n) samps=${OPTARG}
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
coll_script="${base_dir}/src_py/collate_tagged_data.py"

### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

distlists=$(cat ${distlists_fpath})
echo "Collating match data from ${samps} bootstrap-tagged phoms in homology dimension ${match_homdim}."
printf '\n\n'

for distlist in ${distlists}
do
    	for distname in $(cat $distlist)
    	do
		data_label=$(basename ${distname} | cut -d. -f 1)

		# Name the submission script for this dataset type
		sbatch_fpath=$(dirname ${distname})"/do_collate_${data_label}"

		# Create bookkeeping directories
		outdir=$(dirname ${distname})"/phom_data_${data_label}"
		match_dir="${outdir}/matching"

		vmatch_nametype="${match_dir}/verbose_match_dim${match_homdim}_[tagspot].txt"
	
		# collate outputs across group of tags
		###############
		python3 ${coll_script} -f ${vmatch_nametype} -t ${tagfile} -c ${samps} -d "dictlist"
		###############
	done
done
