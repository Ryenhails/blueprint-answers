#!/bin/bash
#SBATCH --job-name=stoa_yolo_hr
#SBATCH --partition=gpu-interactive
#SBATCH --gpus=v100:1
#SBATCH --time=03:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --output=/scratch/work/liuz16/stoa_raster/yolo/job_%j.log
# Higher-resolution run, continued from the pilot weights: does a larger input fix window counting?
# Trains, predicts val/E1/E2, tunes thresholds and scores, all in this job. Does NOT touch the system's detector store.
# usage: sbatch cubi_yolo_hr.sbatch <run_name> <imgsz> <batch> <hours>

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/scratch/work/liuz16/stoa_yolo_lib
export YOLO_CONFIG_DIR=/scratch/work/liuz16/stoa_raster/yolo_cfg

RUN=${1:-hr1536}; IMGSZ=${2:-1536}; BATCH=${3:-8}; HOURS=${4:-2.7}
BASE=/scratch/work/liuz16/stoa_raster/yolo
OUT=$BASE/$RUN
DS=/tmp/stoa_yolo_ds_${SLURM_JOB_ID}
SRC=/scratch/elec/t412-aiwo/stoa/src
mkdir -p "$OUT"
trap 'rm -rf "$DS"' EXIT
echo "Start: $(date) on $(hostname) run=$RUN imgsz=$IMGSZ batch=$BATCH hours=$HOURS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $BASE
python $SRC/cubi_yolo.py prep "$DS" && \
python $SRC/cubi_yolo.py train "$DS" "$OUT" 60 "$IMGSZ" "$HOURS" 1.0 "$BATCH" "$BASE/pilot/run/weights/best.pt" && \
python $SRC/cubi_yolo.py predict "$OUT/run/weights/best.pt" "$OUT/preds.json" "$IMGSZ" && \
python $SRC/cubi_yolo.py eval "$OUT/preds.json" > "$OUT/eval_${SLURM_JOB_ID}.txt" 2>&1
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"; echo "End Time: $(date)"
exit $EXIT_CODE
