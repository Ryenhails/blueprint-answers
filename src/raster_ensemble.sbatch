#!/bin/bash
#SBATCH --job-name=stoa_ens
#SBATCH --partition=batch-milan
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/yolo/ens_%j.log
# Gemini full reads of the 380 official-test cubicasa5k drawings (budget-capped), then the pre-registered ensemble test
# (raster_yolo_plan.md section 12). usage: sbatch raster_ensemble.sbatch [budget_usd]
module load mamba
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
RES=/scratch/work/liuz16/stoa_raster/yolo/ensemble_eval_${SLURM_JOB_ID}.txt
cd /scratch/elec/t412-aiwo/stoa/src
echo "Start: $(date) on $(hostname)"
python raster_ensemble.py read "${1:-14.5}" 6
python raster_ensemble.py eval > "$RES"
EXIT_CODE=$?
cat "$RES"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
