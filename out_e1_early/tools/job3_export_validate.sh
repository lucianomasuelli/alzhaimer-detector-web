#!/bin/bash
#SBATCH --job-name=e1-export-3
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:40:00
#SBATCH --output=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job3_%j.out
#SBATCH --error=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job3_%j.err
source /home/lmasuelli/alzhaimer-detector-web/out_e1_early/tools/env_header.sh
echo "== 4. export =="
python scripts/export_head.py "$RES/logs/ablation/ivanova_e1_abl-early/seed_42/model_fold_0.pth" out_e1_early/ \
  --name e1-audio-early --head-dim 128 --dropout 0.4 --layer-range 0 16 \
  --threshold 0.6699999999999999 --max-duration 160
cat out_e1_early/config.json; sha256sum out_e1_early/head.pth "$RES/logs/ablation/ivanova_e1_abl-early/seed_42/model_fold_0.pth"
echo "== 5 y 6 =="; python out_e1_early/tools/validate_export.py
