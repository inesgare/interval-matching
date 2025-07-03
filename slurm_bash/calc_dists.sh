#!/bin/bash

set -o nounset

# script directory
base_dir="interval-matching_bootstrap"

# file containing a list of BR (modality/dimension, feature) specifications.
# see comments below (lines 54-56) for some context on formatting expectations.
spec_list_fname="spec_list.txt"

mem_gb=500
partition="tier2_cpu"
maxtime_str="167:55:00"

while getopts ":b:s:g:p:t:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) spec_list_fname=${OPTARG}
    ;;
    g) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
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

# file containing a list of BR (modality/dimension, feature) specifications.
# see comments below (lines 54-56) for some context on formatting expectations.
spec_list_fpath="${base_dir}/${spec_list_fname}"	

# path to batch-writing bash script (should be fixed, so is given as a constant)
sbatch_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_bash/submit_distmtx_sbatch.sh"

specifications=$(cat ${spec_list_fpath})

for spec_line in ${specifications}
do
	echo "..."
	echo "Submitting 'distance' computations for ${spec_line} within ${base_dir}"

	# NOTE: for within-modality experiments, "modality" may actually hold the dimension number (e.g., modality="d100"),
	# (con) and when a dimension needs to be specified for some modality in a multiple-modality experiment, the modality-dim
	# (con) pair will be stored in "modality" (e.g., spec_line="ICA300_Amps" -> modality="ICA300", feature="Amps")
	modality=$( echo $spec_line | cut -d_ -f 1 )
	feature=$( echo $spec_line | cut -d_ -f 2 )

	${sbatch_fpath} -b ${base_dir} -m ${modality} -f ${feature} -g ${mem_gb} -p ${partition} -t ${maxtime_str}
done
