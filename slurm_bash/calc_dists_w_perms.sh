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

# NEW INPUT STRUCTURES:
perm_num=100
perm_dir="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing/permutations/subject_perms"
perm_type="subject"
overwrite=false

while getopts ":b:g:p:t:s:n:d:T:o:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    g) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    s) spec_list_fname=${OPTARG}
    ;;
    n) perm_num=${OPTARG}
    ;;
    d) perm_dir=${OPTARG}
    ;;
    T) perm_type=${OPTARG}
    ;;
    o) overwrite=${OPTARG}
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

echo "Conducting permutation testing that shuffles data along \'${perm_type}\' axis..."
# NEED DIFFERENT LOGIC FOR HANDLING DIFFERENT PERMUTATION TYPES!
if [ "${perm_type}" = "subject" ]
then
	#subject_type
	for perm_set in $(ls "${perm_dir}/"perm*set* | head -$perm_num)
	do
		echo "Permuting data using permutations from ${perm_set}"
		for spec_line in ${specifications}
		do
			echo "..."
			echo "Submitting 'distance' computations for ${spec_line} within ${base_dir}"

			# NOTE: for within-modality experiments, "modality" may actually hold the dimension number (e.g., modality="d100"),
			# (con) and when a dimension needs to be specified for some modality in a multiple-modality experiment, the modality-dim
			# (con) pair will be stored in "modality" (e.g., spec_line="ICA300_Amps" -> modality="ICA300", feature="Amps")
			modality=$( echo $spec_line | cut -d_ -f 1 )
			feature=$( echo $spec_line | cut -d_ -f 2 )

			${sbatch_fpath} -b ${base_dir} -m ${modality} -f ${feature} -g ${mem_gb} -p ${partition} -t ${maxtime_str} -P true -T ${perm_type} -s ${perm_set} -o ${overwrite}
		done
	done
elif [ "${perm_type}" = "feature" ]
then
	#feature_type
	for perm_seed in $(seq 1 $perm_num)
	do
		echo "Permuting data using permutations generated from \'np.random.default_rng(${perm_seed})\'"
		for spec_line in ${specifications}
		do
			echo "..."
			echo "Submitting 'distance' computations for ${spec_line} within ${base_dir}"

			# NOTE: for within-modality experiments, "modality" may actually hold the dimension number (e.g., modality="d100"),
			# (con) and when a dimension needs to be specified for some modality in a multiple-modality experiment, the modality-dim
			# (con) pair will be stored in "modality" (e.g., spec_line="ICA300_Amps" -> modality="ICA300", feature="Amps")
			modality=$( echo $spec_line | cut -d_ -f 1 )
			feature=$( echo $spec_line | cut -d_ -f 2 )

			${sbatch_fpath} -b ${base_dir} -m ${modality} -f ${feature} -g ${mem_gb} -p ${partition} -t ${maxtime_str} -P true -T ${perm_type} -r ${perm_seed} -o ${overwrite}
		done
	done
fi
