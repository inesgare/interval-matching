#!/bin/sh

set -o nounset

## bookkeeping paths ###
base_dir="interval-matching_bootstrap"
outdir="${base_dir}/phom_analysis/bootstrap_distances/PROFUMO_vs_null"

barX_fpath=""
barY_fpath=""
tagfile="${base_dir}/subsampling/taglist100k_90p_famstruct.txt"
count=100
bsdist_homdim=1
p=2
q=2
use_affinity=true
verbose=true

mem_gb=10
partition="tier2_cpu"
maxtime_str="23:55:00"

while getopts ":x:y:T:n:D:P:Q:a:v:o:m:p:t:" opt; do
  case $opt in
    x) barX_fpath=${OPTARG}
    ;;
    y) barY_fpath=${OPTARG}
    ;;
    T) tagfile=${OPTARG}
    ;;
    n) count=${OPTARG}
    ;;
    D) bsdist_homdim=${OPTARG}
    ;;
    P) p=${OPTARG}
    ;;
    Q) q=${OPTARG}
    ;;
    a) use_affinity=${OPTARG}
    ;;
    v) verbose=${OPTARG}
    ;;
    o) outdir=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
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

### paths to code ###
bsdist_script="${base_dir}/src_py/comp_bootstrap_dists.py"

Yname=$(basename $(dirname $(ls ${barY_fpath})))

Yname=${Yname/"phom_data"/"Y"}
data_label="X_vs_${Yname}"

sbatch_fpath="${outdir}/do_bsdist_${data_label}"
if $use_affinity
then
	run_suffix="-a "
else
	run_suffix=""
fi
if $verbose
then
	run_suffix="${run_suffix}-v"
fi
outpath="${outdir}/Wp_hat_${data_label}.pkl"

echo "\
\
#!/bin/sh

#SBATCH --job-name=bsdist_${data_label}
#SBATCH --output=${outdir}/logs/bsdist_${data_label}.out
#SBATCH --error=${outdir}/logs/bsdist_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem=${mem_gb}gb

bsdist_script=${bsdist_script}

barX_fpath=${barX_fpath}
barY_fpath=${barY_fpath}
count=${count}
tagfile=${tagfile}
dim=${bsdist_homdim}
outpath=${outpath}

echo \"barX_fpath: \\\"\${barX_fpath}\\\"\"
echo \"barY_fpath: \\\"\${barY_fpath}\\\"\"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate stats
echo \"saving results to \${outpath}\"

python \${bsdist_script} -x \${barX_fpath} -y \${barY_fpath} -t \${tagfile} -n \${count} --dim \${dim} -p ${p} -q ${q} -o \${outpath} ${run_suffix}
\
" > "${sbatch_fpath}"  # Overwrite submission script

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
sbatch "${sbatch_fpath}" || { echo "Error submitting job!"; exit 1; } 
