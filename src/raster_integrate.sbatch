#!/bin/bash
#SBATCH --job-name=stoa_integ
#SBATCH --partition=gpu-interactive
#SBATCH --gpus=v100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/scratch/elec/t412-aiwo/chenyang/logs/stoa_integ_%j.out
#SBATCH --error=/scratch/elec/t412-aiwo/chenyang/logs/stoa_integ_%j.err
# Runs after the detector is trained and its thresholds are tuned (cubi_yolo.sbatch + eval):
#   1. detect on all 4,787 cubicasa5k drawings -> data/raster_detector.jsonl
#   2. choose a reader per kind on held-out drawings -> data/raster_sources.json
#   3. score the dev set through the integrated image route and the whole system
# usage: sbatch --dependency=afterok:<eval job> raster_integrate.sbatch [run_name] [imgsz]

module load mamba
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
source activate r-api-env
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/scratch/work/liuz16/stoa_yolo_lib
export YOLO_CONFIG_DIR=/scratch/work/liuz16/stoa_raster/yolo_cfg

RUN=${1:-pilot}
OUT=/scratch/work/liuz16/stoa_raster/yolo/$RUN
RES=$OUT/integrate_${SLURM_JOB_ID}.txt
cd /scratch/elec/t412-aiwo/stoa/src
{
  echo "Start: $(date) on $(hostname)"
  python raster_detector.py build "$OUT/run/weights/best.pt" "$OUT/preds.thr.json" "${2:-1024}" && \
  python raster_ingest.py select && \
  python raster_ingest.py dev && \
  python system.py dev "$OUT/dev_preds_${SLURM_JOB_ID}.jsonl" --gemini
  echo "Exit Code: $?"
  echo "End Time: $(date)"
} 2>&1 | tee "$RES"
