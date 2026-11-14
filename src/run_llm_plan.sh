#!/bin/bash
#SBATCH --job-name=stoa_plan
#SBATCH --partition=gpu-debug
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --output=/scratch/elec/t412-aiwo/chenyang/logs/stoa_plan_%j.out
#SBATCH --error=/scratch/elec/t412-aiwo/chenyang/logs/stoa_plan_%j.out

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
module load model-huggingface          # shared read-only model cache, offline mode: nothing is downloaded
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "Start: $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# usage: sbatch run_llm_plan.sh <script.py> <args...>
python /scratch/elec/t412-aiwo/stoa/src/"$@"
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
