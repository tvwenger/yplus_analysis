#!/bin/bash
#SBATCH --chdir="/home/twenger/yplus_analysis"
#SBATCH --job-name="combine"
#SBATCH --output="logs/combine_logs/%x.%j.%N.out"
#SBATCH --error="logs/combine_logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-3

eval "$(conda shell.bash hook)"
conda activate mcmc_yplus

tmpdir=`mktemp -d`
echo "starting to combine dataset $SLURM_ARRAY_TASK_ID"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python scripts/combine_results.py $SLURM_ARRAY_TASK_ID
rm -rf $tmpdir
