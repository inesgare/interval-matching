#!/bin/bash

set -o nounset

# script directory
base_dir="interval-matching_bootstrap"

modality=PROFUMO
feature=Amps

mem_gb=500
partition="tier2_cpu"
maxtime_str="167:55:00"

do_perms=false
permute_type="subject"
perm_set=None
perm_seed=0
overwrite=true

while getopts ":b:g:p:t:m:f:P:T:s:r:o:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    g) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    m) modality=${OPTARG}
    ;;
    f) feature=${OPTARG}
    ;;
    P) do_perms=${OPTARG}
    ;;
    T) permute_type=${OPTARG}
    ;;
    s) perm_set=${OPTARG}
    ;;
    r) perm_seed=${OPTARG}
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

subbase_dir="${base_dir}/${modality}_${feature}"
methods_fpath=${subbase_dir}"/methods.csv"
methods=$(cat ${methods_fpath})
# path to source file
pysrc_fpath="<path_to_Gram_matrix_computation_script>"

if [ "${feature}" = "Maps" ]
then
	mem_gb=100
	if [[ "${modality}" == *"100"* ]]
	then
		mem_gb=200
	elif [[ "${modality}" == *"200"* ]]
	then
		mem_gb=400
	elif [[ "${modality}" == *"300"* ]]
	then
		mem_gb=800
	elif [[ "${modality}" == *"PROFUMO"* ]]
	then
		mem_gb=150
	fi
fi

########################## Write the input and the script #########################

for method in ${methods};
do
	echo "Calculating (dis)-similarity using ${method}..."
	# Create scripts
	if [ "${do_perms}" = "true" ]
	then
		if [ "${permute_type}" = "subject" ]
		then
    			perm_name=$(basename "${perm_set}" | cut -d. -f 1)
		elif [ "${permute_type}" = "feature" ]
		then
    			perm_name="seed${perm_seed}"
		else
    			echo "Encountered invalid permutation type \'${permute_type}\' -- aborting."
    			exit
		fi
		
		perms_basedir="${subbase_dir}/permstrapping"
		
		perm_label="${permute_type}Perms_${perm_name}"
		sbatch_fpath="${perms_basedir}/do_sim_${feature}${modality}${method}_${perm_label}"  # path to batch submission file
		label="dists_${perm_label}"
	else
		perms_basedir="${subbase_dir}"

		sbatch_fpath=${perms_basedir}"/do_sim_"${feature}${modality}${method}  # path to batch submission file
		label="dists"	# 'label' was previously either "sims" or "dists" but "sims" use case is deprecated
	fi
	subj_dist="${perms_basedir}/${modality}_${feature}_${method}_${label}.txt"

	if [ "${overwrite}" = "true" ]
	then
		printf "\noverwrite and resubmit existing batch file. ==> "
	fi
	if [ ! -s "${subj_dist}" ]
	then
		printf "\nresults file does not currently exist. ==> "
	fi
	if [ ! -s "${subj_dist}" ] || [ "${overwrite}" = "true" ]
	then
		echo "condition met to overwrite and resubmit batch file:"
		echo "${sbatch_fpath}"
	fi

	if [ ! -s "${subj_dist}" ] || [ "${overwrite}" = "true" ]
	then
		mkdir -p "${perms_basedir}"
		echo "\
\
#!/bin/sh

#SBATCH --job-name=${modality}${feature}_${method}${label}
#SBATCH --output=${perms_basedir}/logs/${modality}${feature}_${method}${label}.out
#SBATCH --error=${perms_basedir}/logs/${modality}${feature}_${method}${label}.err
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --time=${maxtime_str}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=${mem_gb}gb

subj_list=${subbase_dir}/subj_list_${feature}.csv
subj_dist=${perms_basedir}/${modality}_${feature}_${method}_${label}.txt

printf \"Computing dissimilarity matrix...\n\n\"
printf \"pulling subject data from: \\n\${subj_list}\\n\"
printf \"sample data paths:\\n\$(cat \${subj_list} | head -3)\\n...\\n\\n\"
printf \"saving dissimilarity matrix to: \\n\${subj_dist}\\n\"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro

\
		" > "${sbatch_fpath}"  # Overwrite submission script

		if [ "${do_perms}" = "true" ];
		then
			echo "perm_set=${perm_set}" >> ${sbatch_fpath}
			echo "perm_seed=${perm_seed}" >> ${sbatch_fpath}
			echo "" >> ${sbatch_fpath}
			echo "python3 ${pysrc_fpath} -i \${subj_list} -o \${subj_dist} -m ${method} -D -P -t ${permute_type} -s \${perm_set} -r \${perm_seed}" >> ${sbatch_fpath}
		else
			echo "python3 ${pysrc_fpath} -i \${subj_list} -o \${subj_dist} -m ${method} -D" >> ${sbatch_fpath}
		fi

		echo "" >> ${sbatch_fpath}
		echo "echo \${subj_dist} >> ${perms_basedir}/distlist_${feature}${modality}.csv" >> ${sbatch_fpath}

		# Make script executable
		chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

		# Submit script
		sbatch "${sbatch_fpath}" --nice 100000 || { echo "Error submitting jobs!"; exit 1; }
	fi
done
