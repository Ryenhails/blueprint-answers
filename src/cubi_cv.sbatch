#!/bin/bash
#SBATCH --job-name=stoa_cv
#SBATCH --partition=gpu-interactive
#SBATCH --gpus=v100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/yolo/cv_%A_%a.log
# Cross-fitted detectors (raster_yolo_plan.md section 13), one array task per fold:
#   smoke: sbatch --array=0 --time=00:25:00 cubi_cv.sbatch cv_smoke 1
#   main : sbatch --array=0-4 cubi_cv.sbatch cv 80
module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/scratch/work/liuz16/stoa_yolo_lib
export YOLO_CONFIG_DIR=/scratch/work/liuz16/stoa_raster/yolo_cfg
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

RUN=${1:-cv}; EPOCHS=${2:-80}; FOLD=${SLURM_ARRAY_TASK_ID:-0}
OUT=/scratch/work/liuz16/stoa_raster/yolo/$RUN/fold$FOLD
DS=/tmp/stoa_cv_ds_${SLURM_JOB_ID}
SRC=/scratch/elec/t412-aiwo/stoa/src
mkdir -p "$OUT"
trap 'rm -rf "$DS" "$TRITON_CACHE_DIR"' EXIT
echo "Start: $(date) on $(hostname) run=$RUN fold=$FOLD epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /scratch/work/liuz16/stoa_raster/yolo     # yolo11n.pt here serves ultralytics' AMP check offline
python $SRC/cubi_cv.py prep "$DS" "$FOLD" && \
python $SRC/cubi_cv.py train "$DS" "$OUT" "$EPOCHS" && \
python $SRC/cubi_cv.py predict "$OUT/run/weights/last.pt" "$OUT/preds.json" "$FOLD"
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
