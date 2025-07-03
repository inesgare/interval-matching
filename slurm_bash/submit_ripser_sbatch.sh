#!/bin/sh

set -o nounset

dim=2
do_img=true
mem_gb=7
ldmZ_fpath=""
partition="tier2_cpu"
data_label="test"
maxtime_str="23:55:00"
ripser_fpath="interval-matching_bootstrap/modified_ripser/ripser-image-persistence-simple/ripser-image"

while getopts ":x:z:f:o:d:i:m:p:t:D:r:s:" opt; do
  case $opt in
    x) ldmX_fpath=${OPTARG}
    ;;
    z) ldmZ_fpath=${OPTARG}
    ;;
    f) sbatch_fpath=${OPTARG}
    ;;
    o) outpath=${OPTARG}
    ;;
    d) data_label=${OPTARG}
    ;;
    i) do_img=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    D) dim=${OPTARG}
    ;;
    r) ripser_fpath=${OPTARG}
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

if $do_img; then
	sbatch_fpath="${sbatch_fpath}_img"
fi

script_dir=$( dirname $sbatch_fpath )

job_name="shbsphoms_${data_label}"
logdir="${script_dir}/logs"
if ! compgen -G $logdir >> /dev/null
then
	mkdir $logdir
fi
logpath="${logdir}/${job_name}"

# at one point, these were the nodes that regularly failed and had to be excluded -- the list is currently shorter, i hope?
   ###SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07
echo "\
\
#!/bin/sh

#SBATCH --exclude=node15,node23,node21,node25,node27
#SBATCH --job-name=${job_name}
#SBATCH --output=${logpath}.out%j
#SBATCH --error=${logpath}.err%j
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem=${mem_gb}gb

dim=${dim}
ripser_fpath=\"${ripser_fpath}\"

ldmX_fpath=\"${ldmX_fpath}\"
ldmZ_fpath=\"${ldmZ_fpath}\"

outpath=\"${outpath}\"

echo \"input_ldm_fpath: \\\'\${ldmX_fpath}\\\'\"

\
" > "${sbatch_fpath}"  # Overwrite submission script

# slightly different function calls/parameters depending on if we are doing image persistence
if $do_img
then
    echo \"image_ldm_fpath: \\\'\${ldmZ_fpath}\\\'\" >> "${sbatch_fpath}"
    if [[ -z ${ldmZ_fpath:+.} ]]
    then
        echo "ldmZ_fpath must be set when doing ripser-image"
        exit 1
    fi
    echo "\
\${ripser_fpath} --dim \${dim} --subfiltration \${ldmX_fpath} \${ldmZ_fpath} >> \${outpath}
    " >> "${sbatch_fpath}"
else
    echo "\
\${ripser_fpath} --dim \${dim} \${ldmX_fpath} >> \${outpath}
    " >> "${sbatch_fpath}"
fi

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
echo "Submitting ${sbatch_fpath} and writing phom data to ${outpath} "
sbatch "${sbatch_fpath}" --nice 200000 || { echo "Error submitting jobs!"; exit 1; }
echo ""
