#!/bin/bash
#SBATCH --job-name=stoa_cubi
#SBATCH --partition=gpu-debug
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/cubi_infer_%j.log

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1

echo "Start: $(date)  node: $(hostname)"
STOA_NOPOLY=1 python /scratch/elec/t412-aiwo/stoa/src/raster_cubicasa_infer.py 5000 1620
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
