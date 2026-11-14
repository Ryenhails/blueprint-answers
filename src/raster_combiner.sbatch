#!/bin/bash
#SBATCH --job-name=stoa_comb
#SBATCH --partition=batch-milan
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/combiner/job_%j.log
# usage: sbatch raster_combiner.sbatch read <budget_usd> <workers>   |   sbatch raster_combiner.sbatch fit [cv|pilot] [train_roles]
module load mamba
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
RES=/scratch/work/liuz16/stoa_raster/combiner/$1_${SLURM_JOB_ID}.txt
cd /scratch/elec/t412-aiwo/stoa/src
echo "Start: $(date) on $(hostname): $*"
python raster_combiner.py "$@" 2>&1 | tee "$RES"
EXIT_CODE=${PIPESTATUS[0]}
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
