#!/bin/bash
#SBATCH --job-name=stoa_yolo
#SBATCH --partition=gpu-interactive
#SBATCH --gpus=v100:1
#SBATCH --time=03:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/yolo/job_%j.log
# usage: sbatch [--time=00:20:00] cubi_yolo.sbatch <run_name> <epochs> <hours> [fraction]
#   smoke: sbatch --time=00:20:00 cubi_yolo.sbatch smoke 1 0.1 0.05
#   pilot: sbatch cubi_yolo.sbatch pilot 60 2.6

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/scratch/work/liuz16/stoa_yolo_lib
export YOLO_CONFIG_DIR=/scratch/work/liuz16/stoa_raster/yolo_cfg
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

RUN=${1:-smoke}; EPOCHS=${2:-1}; HOURS=${3:-0.1}; FRAC=${4:-1.0}
OUT=/scratch/work/liuz16/stoa_raster/yolo/$RUN
DS=/tmp/stoa_yolo_ds_${SLURM_JOB_ID}          # node-local disk: ~4,500 images + labels never touch the project quota
SRC=/scratch/elec/t412-aiwo/stoa/src
mkdir -p "$OUT"
trap 'rm -rf "$DS" "$TRITON_CACHE_DIR"' EXIT

echo "Start: $(date) on $(hostname) run=$RUN epochs=$EPOCHS hours=$HOURS fraction=$FRAC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /scratch/work/liuz16/stoa_raster/yolo     # yolo11n.pt here is used by ultralytics' AMP check (no download)
python $SRC/cubi_yolo.py prep "$DS" && \
python $SRC/cubi_yolo.py train "$DS" "$OUT" "$EPOCHS" 1024 "$HOURS" "$FRAC" && \
python $SRC/cubi_yolo.py predict "$OUT/run/weights/best.pt" "$OUT/preds.json" 1024
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
