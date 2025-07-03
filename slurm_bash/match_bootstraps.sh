#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="interval-matching_bootstrap"
subbase_dir="${base_dir}/bootstrap_benchmarks"
tagfile="${base_dir}/taglist.txt"

### input datasets ###
distlists_fpath="${subbase_dir}/real_distlists.csv"

### run parameters ###
samps=3
match_homdim=1
mem_gb=10

### argument parsing ###
while getopts ":b:s:f:t:n:D:m:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) subbase_dir=${OPTARG}
    ;;
    f) distlists_fpath=${OPTARG}
    ;;
    t) tagfile=${OPTARG}
    ;;
    n) samps=${OPTARG}
    ;;
    D) match_homdim=${OPTARG}
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
extract_src="${base_dir}/utils_match/extract.py"
match_src="${base_dir}/src_bash/submit_match_sbatch.sh"

### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

distlists=$(cat ${distlists_fpath})
tags=""
xtr_only=false

if [ -s $tagfile ] && [[ "${samps}" -gt "0" ]]
then
	echo "Matching cycles in ${samps} phoms in homology dimension ${match_homdim}. Pulling from ${distlists_fpath}."
	printf '\n\n'
	tags=$( cat ${tagfile} | head -$samps )
elif [[ "${samps}" -eq "0" ]]
then
	echo "Specified that samps=${samps} bootstraps are to be used; skipping matching and proceeding with extraction only." 
	xtr_only=true
else
	echo "Error: no tagfile given"
	exit
fi

for distlist in ${distlists}
do
	echo ""
	echo "Iterating through ${distlist}..."
    	for distname in $(cat $distlist)
    	do
		data_label=$(basename ${distname} | cut -d. -f 1)

		# Name the submission script for this dataset type
		sbatch_fpath=$(dirname ${distname})"/do_match_${data_label}"

		# Create bookkeeping directories
		outdir=$(dirname ${distname})"/phom_data_${data_label}"
		phomdir="${outdir}/phom_out"
		matchdir="${outdir}/matching"
		mkdir -p $phomdir
		mkdir -p $matchdir

		# Extract bar and tightrep indices from Ripser output
		phomX_fpath="${outdir}/phom_X.txt"
		if [ -s ${phomX_fpath/phom_X/bars_X} ]
		then
			echo "Skipping extraction. Bars output file already exists:"
			ls ${phomX_fpath/phom_X/bars_X}
		else
			python ${extract_src} -x ${phomX_fpath} -0 -w 
			if $xtr_only
			then
				echo "extracting from:"
				ls ${phomX_fpath}
			fi
		fi
		if $xtr_only
		then
			continue
		fi

    		for tag in $tags
		do
			# name the (generic type of) output file for persistent homology data
			phomY_fpath=${phomdir}"/phomY_${tag}.txt"
			if ! [ -s ${phomY_fpath} ]
			then
				echo "Error! File does not exist:"
				echo ${phomY_fpath}
				continue
			fi

			# extract and write diagram summaries from Ripser output
			python ${extract_src} -x ${phomY_fpath} -0 -w
			python ${extract_src} -x ${phomY_fpath/phomY/phomXZ} -0 -w -i
			python ${extract_src} -x ${phomY_fpath/phomY/phomYZ} -0 -w -i 
		
			# script submitter for matching job
			${match_src} -x ${phomX_fpath} -y ${phomY_fpath} -D ${match_homdim} -f ${sbatch_fpath} -d ${data_label} -m ${mem_gb} -s ${subbase_dir}
		done
	done
done
