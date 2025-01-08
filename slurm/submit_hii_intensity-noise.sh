#!/bin/bash
#SBATCH --chdir="/home/twenger/yplus_analysis"
#SBATCH --job-name="hii_intensity-noise"
#SBATCH --output="logs/hii_intensity-noise_logs/%x.%j.%N.out"
#SBATCH --error="logs/hii_intensity-noise_logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-624%175

# 1250 spectra = 625 jobs of 2 spectra, limit to 175 tasks to limit resource usage
DIR_NAME="hii_intensity-noise"
PER_JOB=2
NUM_SPEC=1250
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $PER_JOB ))
END_IDX=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $PER_JOB ))

eval "$(conda shell.bash hook)"
conda activate bayes_yplus

for (( idx=$START_IDX; idx<$END_IDX; idx++ )); do
    if [ $idx -ge $NUM_SPEC ]; then
	    break
    fi

    # temporary pytensor compiledir
    tmpdir=`mktemp -d`
    echo "starting to analyze $idx"
    PYTENSOR_FLAGS="base_compiledir=$tmpdir" python scripts/fit.py $DIR_NAME $idx slurm
    rm -rf $tmpdir
done
