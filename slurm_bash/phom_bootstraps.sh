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
bootstrap_prop=0.8
maxhomdim=1
do_X_phom=true
mem_gb=300

### argument parsing ###
while getopts ":b:s:f:t:n:p:D:x:m:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) subbase_dir=${OPTARG}
    ;;
    f) distlists_fpath=${OPTARG}
    ;;
    t) tagpath=${OPTARG}
    ;;
    n) samps=${OPTARG}
    ;;
    p) bootstrap_prop=${OPTARG}
    ;;
    D) maxhomdim=${OPTARG}
    ;;
    x) do_X_phom=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
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
# Base version of Ripser:
ripser_rep_path="interval-matching_bootstrap/modified_ripser/ripser-tight-representative-cycles/ripser-representatives"

# Ripser variant for image persistence (default option):
ripser_img_path="interval-matching_bootstrap/modified_ripser/ripser-image-persistence-simple/ripser-image"

# Script to calcluate and save lower distance matrices of bootstrapped spaces
make_ldm_images="${base_dir}/utils_match/make_ldm_images.py"

# Batch job submission script 
ripser_scripter="${base_dir}/src_bash/submit_ripser_sbatch.sh"

# Path to script generating "tags" (bit-encoded subampling indices) if prefab list not provided
gentags_script="${base_dir}/src_py/generate_subindex.py"



### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

echo "Running bash-anarchic benchmarking on: "${distlists_fpath}
distlists=$(cat ${distlists_fpath})

ndims=1003
if compgen -G $tagpath >> /dev/null
then
	echo "Running ${samps} bootstraps from taglist \"${tagpath}\" with maximum homology dimension ${maxhomdim} -- first and last tags are as follows:"
	cat ${tagpath} | head -3
	echo "..."
	cat ${tagpath} | head -$samps | tail -3
	tags=$( cat ${tagpath} | head -$samps )
else
	echo "Running ${samps} bootstraps at a sampling proportion of ${bootstrap_prop} with maximum homology dimension ${maxhomdim}."
	tags=$(python ${gentags_script} --count ${samps} --dims ${ndims} --proportion ${bootstrap_prop})
	for tag in $tags
	do
		echo $tag >> $tagpath
	done
fi

echo "Do X phom: ${do_X_phom}"
printf '\n\n'

for distlist in ${distlists}
do
	echo ""
	echo "Pulling from ${distlist}"
    	for distname in $(cat $distlist)
    	do
		echo "Currently bootstrapping data for ${distname}"
		data_label=$(basename ${distname} | cut -d. -f 1)

		# Name the submission script for this dataset type
		sbatch_fpath=$(dirname ${distname})"/do_phoms_${data_label}"

		# Create bookkeeping directories
		outdir=$(dirname ${distname})"/phom_data_${data_label}"
		ldm_dir="${outdir}/dist_mtxs"
		mkdir -p $ldm_dir
		phomdir="${outdir}/phom_out"
		mkdir -p $phomdir

		dX_fpath="${ldm_dir}/dX.ldm"
		if ! [ -s ${dX_fpath} ]
		then
			cp $distname $dX_fpath
		fi
		ndims=$(wc -l ${dX_fpath})	# to avoid "unset variable"-type error in cases that do not trigger earlier if statements

		phomX_outpath="${outdir}/phom_X.txt"
		if ${do_X_phom} & ! [ -s ${phomX_outpath} ]
		then
			# submit persistence job for dX
			${ripser_scripter} -x ${dX_fpath} -f ${sbatch_fpath} -o ${phomX_outpath} -r ${ripser_rep_path} -i false -m 50 -d ${data_label}"_X" -D ${maxhomdim}
		fi

    		for tag in $tags
		do
			# compute dZ_(tag) and dY_(tag) from dX and subsample indices (via `_subsamp_dZ` subfunction of `make_ldm_images.py')
			# make dXZ_(tag) and dYZ_(tag) (via `make_ldm_images.py`)
			dY_fpath=${ldm_dir}"/dY_${tag}.ldm"
			dXZ_fpath=${ldm_dir}"/dXZ_${tag}.ldm"
			dYZ_fpath=${ldm_dir}"/dYZ_${tag}.ldm"
			dZ_fpath=${ldm_dir}"/dZ_${tag}.ldm"
			if ! test -s $dZ_fpath
			then
				python ${make_ldm_images} -x ${dX_fpath} -t "${tag}" -z ${dZ_fpath} -y ${dY_fpath} -i ${dXZ_fpath} -j ${dYZ_fpath} || \
					printf "Failed to parse arguments to make_ldm_images (likely unreadable tag): \n${tag}\n\n" continue
			fi

			# name the (generic type of) output path for persistent homology data
			phom_outpath=${phomdir}"/phomXZ_"${tag}".txt"

			# submit image persistence job for dXZ_(tag) and dYZ_(tag) --in style of `do_ripser_test` sbatch script
			if ! test -s $phom_outpath
			then
				${ripser_scripter} -x ${dXZ_fpath} -z ${dZ_fpath} -f ${sbatch_fpath} -o ${phom_outpath} -m ${mem_gb} -d "${data_label}_XZ" -D ${maxhomdim}
			fi
			if ! test -s ${phom_outpath/phomXZ/phomYZ} 
			then
				${ripser_scripter} -x ${dYZ_fpath} -z ${dZ_fpath} -f ${sbatch_fpath} -o ${phom_outpath/phomXZ/phomYZ} -m ${mem_gb} -d "${data_label}_YZ" -D ${maxhomdim}
			fi

			# submit persistence job for dY_(tag)
			if ! test -s ${phom_outpath/phomXZ/phomY} 
			then
				${ripser_scripter} -x ${dY_fpath} -f ${sbatch_fpath} -o ${phom_outpath/phomXZ/phomY} -r ${ripser_rep_path} -i false -d "${data_label}_Y" -D ${maxhomdim}
			fi
		done
	done
done
