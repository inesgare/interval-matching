base_dir=${1:-"interval-matching_bootstrap"}
src_fpath=${2:-"interval-matching_bootstrap/visualization/prevwt_PD.py"}
dim=${3:-1}
num_bootstraps=${4:-1000}

out_dir="$( dirname ${src_fpath} )/prev-weighted_persistence_diagrams"

if ! compgen -G ${out_dir} >> /dev/null
then
	mkdir -p ${out_dir}
fi

prev_fpaths=$( ls ${base_dir}/within_*/*/phom_data*/prev* )
for prev_fpath in ${prev_fpaths}
do
	parent_dir=$( dirname ${prev_fpath} )
	echo "Plotting weighted PDs in ${parent_dir}..."
	modality_name=$( basename "$( dirname $parent_dir )" )
	distance_name=$( basename $parent_dir | cut -d_ -f 5 )
	label="${modality_name}_${distance_name}"

	bars_fpath="${parent_dir}/bars_X.txt"
	fig_outpath="${out_dir}/prevwt_PD_${label}.png"

	echo "bars from: ${bars_fpath}"
	echo "prevalence from: ${prev_fpath}"
	
	python ${src_fpath} -b "${bars_fpath}" -p "${prev_fpath}" -f "${fig_outpath}" -l "${label}"
	
	echo ""
done
