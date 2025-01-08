#!/bin/bash
#SBATCH --chdir="/home/twenger/yplus_analysis"
#SBATCH --job-name="dig_noise_1.0"
#SBATCH --output="logs/dig_noise_1.0_logs/%x.%j.%N.out"
#SBATCH --error="logs/dig_noise_1.0_logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-999%180

# 112540 spectra = 1000 jobs of 113 spectra, limit to 180 tasks to limit resource usage
DIR_NAME="dig_noise_1.0"
PER_JOB=113
NUM_SPEC=112540
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

