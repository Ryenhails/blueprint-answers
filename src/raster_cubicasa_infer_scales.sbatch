#!/bin/bash
#SBATCH --job-name=stoa_cubs
#SBATCH --partition=gpu-debug
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/cubi_scales_%j.log

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1

echo "Start: $(date)  node: $(hostname)"
for S in 0.5 0.75 1.5 2 3; do STOA_SCALE=$S python /scratch/elec/t412-aiwo/stoa/src/raster_cubicasa_infer.py 300 300; done
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
