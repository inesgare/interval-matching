#!/bin/sh

set -o nounset

## bookkeeping paths ###
base_dir="interval-matching_bootstrap"
outdir="${base_dir}/phom_analysis/bootstrap_distances/PROFUMO_vs_null"

barsX_fpath=""
listPerm_fpath=""
permdist_homdim=1
p=2
q=2
verbose=true

mem_gb=10
partition="tier2_cpu"
maxtime_str="23:55:00"

while getopts ":x:y:D:P:Q:v:o:m:p:t:" opt; do
  case $opt in
    x) barsX_fpath=${OPTARG}
    ;;
    y) listPerm_fpath=${OPTARG}
    ;;
    D) permdist_homdim=${OPTARG}
    ;;
    P) p=${OPTARG}
    ;;
    Q) q=${OPTARG}
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
permdist_script="${base_dir}/src_py/comp_permtest_dists.py"

Permname=$(basename $(dirname $(ls ${listPerm_fpath})))

Permname=${Permname/"phom_data"/"Perm"}
data_label="X_vs_${Permname}"

sbatch_fpath="${outdir}/do_permdist_${data_label}"
if $verbose
then
	run_suffix="-v "
fi

echo "\
\
#!/bin/sh

#SBATCH --job-name=permdist_${data_label}
#SBATCH --output=${outdir}/logs/permdist_${data_label}.out
#SBATCH --error=${outdir}/logs/permdist_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem=${mem_gb}gb

permdist_script=${permdist_script}

barsX_fpath=${barsX_fpath}
listPerm_fpath=${listPerm_fpath}
dim=${permdist_homdim}
outdir=${outdir}

echo \"barsX_fpath: \\\"\${barsX_fpath}\\\"\"
echo \"listPerm_fpath: \\\"\${listPerm_fpath}\\\"\"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate stats
echo \"saving results in \${outdir}\"

python \${permdist_script} -x \${barsX_fpath} -y \${listPerm_fpath} --dim \${dim} -p ${p} -q ${q} -o \${outdir} ${run_suffix}
\
" > "${sbatch_fpath}"  # Overwrite submission script

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
sbatch "${sbatch_fpath}" || { echo "Error submitting job!"; exit 1; } 
